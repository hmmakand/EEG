"""Shared, configuration-fingerprinted CFSPMNet LOSO experiment runner."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import braindecode
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from eeg_bci.cfspmnet.canonicalization import load_liu2024_participants
from eeg_bci.cfspmnet.model import CFSPMNet
from eeg_bci.cfspmnet.preprocessing import (
    PAPER_ALIGNED_FIGSHARE,
    PROFILE_IMPLEMENTATION_VERSIONS,
    PROJECT_MOABB,
    PreprocessingName,
    load_trials_for_profile,
)
from eeg_bci.cfspmnet.protocols import (
    AFFECTED_UNAFFECTED_FLIP,
    CLASS_NAMES_BY_CANONICALIZATION,
    RAW_LEFT_RIGHT,
    SOURCE_ONLY,
    SPPM_TRANSDUCTIVE,
    CanonicalizationMode,
    TrainingProtocol,
    build_full_target_loso_fold,
    canonicalize_trial_arrays,
)
from eeg_bci.cfspmnet.sppm import (
    build_shared_private_signature_prototypes,
    compute_private_signature_features,
)
from eeg_bci.cfspmnet.training import (
    SPPMTrainingConfig,
    build_sppm_dataloaders,
    evaluate_target,
    initialize_target_pseudo_labels,
    train_source_only_epoch,
    train_sppm_epoch,
)
from eeg_bci.tracking.artifacts import (
    prepare_run_dirs,
    save_dataset_info,
    save_final_metrics,
)
from eeg_bci.tracking.metrics import summarize_scalar_metrics
from eeg_bci.tracking.naming import held_out_label, timestamp as make_timestamp
from eeg_bci.tracking.results import append_master_result
from eeg_bci.tracking.run_recording import resolve_device
from eeg_bci.tracking.tensorboard import (
    write_confusion_matrix,
    write_final_scalars,
)
from eeg_bci.utils.seed import seed_everything

logger = logging.getLogger(__name__)

DATASET_NAME = "liu2024"
METHOD = "leave_one_subject_out"
RECIPE_VERSION = "cfspmnet-comparison-v2"
OptimizerName = Literal["adam", "adamw"]
SchedulerName = Literal["none", "cosine"]
SCIENTIFIC_MODULE_FILES = (
    "canonicalization.py",
    "datasets.py",
    "experiment.py",
    "frsm.py",
    "model.py",
    "preprocessing.py",
    "protocols.py",
    "sppm.py",
    "tokenizer.py",
    "training.py",
)
SCIENTIFIC_SHARED_PROJECT_FILES = (
    "src/eeg_bci/data/types.py",
    "src/eeg_bci/tracking/metrics.py",
    "src/eeg_bci/utils/seed.py",
)
SCIENTIFIC_MOABB_FILES = (
    "src/eeg_bci/data/moabb.py",
    "src/eeg_bci/data/paths.py",
    "src/eeg_bci/data/preprocessing.py",
    "src/eeg_bci/data/windowing.py",
)


@dataclass(frozen=True)
class CFSPMExperimentConfig:
    experiment_name: str
    model_name: str
    protocol: TrainingProtocol
    canonicalization: CanonicalizationMode
    preprocessing: PreprocessingName
    subject_ids: list[int] = field(default_factory=lambda: list(range(1, 51)))
    held_out_subject_ids: list[int] | None = None
    stage_i_epochs: int = 25
    stage_ii_epochs: int = 175
    alpha: float = 0.98
    pseudo_threshold: float = 0.60
    matching_tolerance_floor: float = 0.50
    batch_size: int = 16
    learning_rate: float = 0.000625
    weight_decay: float = 0.0
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "cosine"
    num_workers: int = 0
    seed: int = 2
    device: str = "auto"
    resume: bool = True
    apply_ica: bool = True
    download_if_missing: bool = True
    figshare_cache_dir: Path | None = None
    output_root: Path | None = None
    model_hparams: dict[str, Any] = field(
        default_factory=lambda: {
            "emb_size": 30,
            "depth": 2,
            "eeg_f1": 8,
            "rhythm_sparsity_threshold": 0.01,
            "rhythm_low_ratio": 0.45,
        }
    )
    recipe_version: str = RECIPE_VERSION

    @property
    def total_epochs(self) -> int:
        return self.stage_i_epochs + self.stage_ii_epochs

    @property
    def effective_held_out_subject_ids(self) -> list[int]:
        return (
            list(self.subject_ids)
            if self.held_out_subject_ids is None
            else list(self.held_out_subject_ids)
        )


def validate_experiment_config(config: CFSPMExperimentConfig) -> None:
    if len(config.subject_ids) < 2 or len(set(config.subject_ids)) != len(config.subject_ids):
        raise ValueError("subject_ids must contain at least two unique cohort subjects.")
    invalid = [value for value in config.subject_ids if value not in range(1, 51)]
    if invalid:
        raise ValueError(f"Liu2024 cohort subject IDs must be in 1..50; got {invalid}.")
    held_out = config.effective_held_out_subject_ids
    if not held_out or len(set(held_out)) != len(held_out):
        raise ValueError("held_out_subject_ids must be a non-empty unique list.")
    missing = sorted(set(held_out) - set(config.subject_ids))
    if missing:
        raise ValueError(f"Held-out subjects are not in the cohort: {missing}.")
    if config.total_epochs <= 0 or config.stage_i_epochs < 0 or config.stage_ii_epochs < 0:
        raise ValueError("Stage epoch counts must be non-negative with a positive total.")
    if config.protocol == SPPM_TRANSDUCTIVE and (
        config.stage_i_epochs == 0 or config.stage_ii_epochs == 0
    ):
        raise ValueError("SPPM requires positive Stage-I and Stage-II epoch counts.")
    if config.protocol not in {SOURCE_ONLY, SPPM_TRANSDUCTIVE}:
        raise ValueError(f"Unsupported protocol: {config.protocol!r}.")
    if config.canonicalization not in {
        RAW_LEFT_RIGHT,
        AFFECTED_UNAFFECTED_FLIP,
    }:
        raise ValueError(f"Unsupported canonicalization: {config.canonicalization!r}.")
    if config.preprocessing not in {PROJECT_MOABB, PAPER_ALIGNED_FIGSHARE}:
        raise ValueError(f"Unsupported preprocessing: {config.preprocessing!r}.")
    if (
        config.preprocessing == PROJECT_MOABB
        and config.protocol == SPPM_TRANSDUCTIVE
        and config.canonicalization != AFFECTED_UNAFFECTED_FLIP
    ):
        raise ValueError(
            "Project-profile SPPM requires affected/unaffected canonicalization."
        )
    if config.preprocessing == PAPER_ALIGNED_FIGSHARE and (
        config.protocol != SPPM_TRANSDUCTIVE
        or config.canonicalization != RAW_LEFT_RIGHT
    ):
        raise ValueError(
            "The paper-aligned profile requires raw left/right transductive SPPM."
        )
    if config.optimizer not in {"adam", "adamw"}:
        raise ValueError(f"Unsupported optimizer: {config.optimizer!r}.")
    if config.scheduler not in {"none", "cosine"}:
        raise ValueError(f"Unsupported scheduler: {config.scheduler!r}.")
    if config.batch_size <= 0 or config.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")
    if not 0.0 <= config.alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    if not 0.0 <= config.pseudo_threshold <= 1.0:
        raise ValueError("pseudo_threshold must lie in [0, 1].")
    if not -1.0 <= config.matching_tolerance_floor <= 1.0:
        raise ValueError("matching_tolerance_floor must lie in [-1, 1].")


def scientific_config_fingerprint(config: CFSPMExperimentConfig) -> str:
    """Hash result-defining settings, configs, and implementation sources."""

    payload = asdict(config)
    for operational_key in (
        "held_out_subject_ids",
        "resume",
        "device",
        "download_if_missing",
        "figshare_cache_dir",
        "output_root",
    ):
        payload.pop(operational_key, None)
    payload["_preprocessing_implementation_version"] = (
        PROFILE_IMPLEMENTATION_VERSIONS[config.preprocessing]
    )
    payload["_scientific_library_versions"] = _scientific_library_versions()
    payload["_scientific_source_sha256"] = _scientific_source_digests(config)
    normalized = _jsonable(payload)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _scientific_source_digests(
    config: CFSPMExperimentConfig,
) -> dict[str, str]:
    repo_root = _find_repo_root(Path(__file__))
    package_dir = Path(__file__).resolve().parent
    paths = [package_dir / name for name in SCIENTIFIC_MODULE_FILES]
    paths.extend(
        repo_root / relative_path
        for relative_path in SCIENTIFIC_SHARED_PROJECT_FILES
    )
    if config.preprocessing == PAPER_ALIGNED_FIGSHARE:
        paths.append(package_dir / "figshare_liu2024.py")
    if config.preprocessing == PROJECT_MOABB:
        paths.extend(
            repo_root / relative_path
            for relative_path in SCIENTIFIC_MOABB_FILES
        )
        paths.extend(
            [
                repo_root / "configs" / "dataset" / "liu2024.yaml",
                repo_root / "configs" / "preprocessing" / "liu2024.yaml",
            ]
        )
    return {
        str(path.relative_to(repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _scientific_library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "braindecode",
        "mne",
        "moabb",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def run_cfspmnet_experiment(
    config: CFSPMExperimentConfig,
) -> list[dict[str, Any]]:
    """Run pending folds and append project-standard master rows."""

    validate_experiment_config(config)
    fingerprint = scientific_config_fingerprint(config)
    repo_root = _find_repo_root(Path(__file__))
    output_root = _resolve_output_root(repo_root, config.output_root)
    results_path = _master_result_path(
        output_root,
        experiment=config.experiment_name,
    )
    completed = (
        _completed_subjects(
            results_path,
            model_name=config.model_name,
            fingerprint=fingerprint,
        )
        if config.resume
        else set()
    )
    pending = [
        subject_id
        for subject_id in config.effective_held_out_subject_ids
        if subject_id not in completed
    ]
    if not pending:
        logger.info(
            "All requested folds already completed for configuration %s.",
            fingerprint,
        )
        return []

    device = resolve_device(config.device)
    loaded = load_trials_for_profile(
        repo_root=repo_root,
        subject_ids=list(config.subject_ids),
        profile_name=config.preprocessing,
        seed=config.seed,
        apply_ica=config.apply_ica,
        download_if_missing=config.download_if_missing,
        figshare_cache_dir=config.figshare_cache_dir,
    )

    participants = None
    if config.canonicalization == AFFECTED_UNAFFECTED_FLIP:
        participants = load_liu2024_participants(
            _participants_path(
                repo_root,
                config.figshare_cache_dir,
                prefer_figshare=(
                    config.preprocessing == PAPER_ALIGNED_FIGSHARE
                ),
            )
        )
    x, y = canonicalize_trial_arrays(
        loaded.x,
        loaded.y,
        loaded.subject_ids,
        mode=config.canonicalization,
        channel_names=loaded.channel_names,
        participants=participants,
    )
    signatures = None
    if config.protocol == SPPM_TRANSDUCTIVE:
        signatures = compute_private_signature_features(
            x,
            channel_names=loaded.channel_names,
        )

    run_name = (
        f"{make_timestamp()}__seed{config.seed}__cfg{fingerprint}"
    )
    run_root = (
        output_root
        / "runs"
        / DATASET_NAME
        / config.experiment_name
        / METHOD
        / config.model_name
        / run_name
    )
    tb_root = (
        output_root
        / "tensorboard"
        / DATASET_NAME
        / config.experiment_name
        / METHOD
        / config.model_name
        / run_name
    )
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = run_root / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            _jsonable(
                {
                    "configuration": asdict(config),
                    "config_fingerprint": fingerprint,
                    "preprocessing": loaded.preprocessing_metadata,
                }
            )
        ),
        config_path,
    )

    rows: list[dict[str, Any]] = []
    for held_out_subject in pending:
        fold_seed = config.seed + int(held_out_subject)
        try:
            seed_everything(fold_seed)
            fold = build_full_target_loso_fold(
                x,
                y,
                loaded.subject_ids,
                held_out_subject=held_out_subject,
                protocol=config.protocol,
                signature_vectors=signatures,
                n_classes=loaded.dataset_info.n_outputs,
            )
            row = _run_one_fold(
                config=config,
                fingerprint=fingerprint,
                fold_seed=fold_seed,
                fold=fold,
                dataset_info=loaded.dataset_info,
                preprocessing_metadata=loaded.preprocessing_metadata,
                device=device,
                run_root=run_root,
                tb_root=tb_root,
                config_path=config_path,
            )
        except Exception:
            logger.exception("Fold for held-out subject %s failed.", held_out_subject)
            row = _base_master_row(
                config=config,
                fingerprint=fingerprint,
                held_out_subject=held_out_subject,
                dataset_info=loaded.dataset_info,
                run_dir=run_root / held_out_label(held_out_subject),
                tb_dir=tb_root / held_out_label(held_out_subject),
                config_path=config_path,
                status="error",
            )
        append_master_result(results_path, row)
        rows.append(row)
        logger.info(
            "held_out_subject=%s status=%s test_accuracy=%s",
            held_out_subject,
            row.get("status"),
            row.get("test_accuracy"),
        )
    return rows


def _run_one_fold(
    *,
    config: CFSPMExperimentConfig,
    fingerprint: str,
    fold_seed: int,
    fold: Any,
    dataset_info: Any,
    preprocessing_metadata: dict[str, Any],
    device: torch.device,
    run_root: Path,
    tb_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    held_out_subject = int(fold.held_out_subject)
    run_dir = run_root / held_out_label(held_out_subject)
    tb_dir = tb_root / held_out_label(held_out_subject)
    run_dirs = prepare_run_dirs(run_dir)
    model = CFSPMNet.from_dataset_info(
        dataset_info,
        **config.model_hparams,
    ).to(device)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)
    epoch_rows: list[dict[str, Any]] = []
    last_train_loss: float | None = None
    stage_i_source_eval_loss: float | None = None
    stage_i_source_eval_accuracy: float | None = None
    stage_i_source_eval_balanced_accuracy: float | None = None
    final_accepted_ratio: float | None = None
    final_confidence_pass_ratio: float | None = None
    final_signature_pass_ratio: float | None = None
    final_active_ratio: float | None = None
    n_target_adapt = 0
    writer = SummaryWriter(log_dir=str(tb_dir))

    try:
        if config.protocol == SOURCE_ONLY:
            source_loader = _classification_loader(
                fold.source_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
            )
            for epoch in range(config.total_epochs):
                last_train_loss = train_source_only_epoch(
                    source_loader,
                    model,
                    optimizer,
                    device,
                )
                _step_scheduler(scheduler)
                learning_rate = float(optimizer.param_groups[0]["lr"])
                epoch_rows.append(
                    {
                        "stage": "source_only",
                        "epoch": epoch,
                        "train_loss": last_train_loss,
                        "source_loss": last_train_loss,
                        "target_loss": None,
                        "accepted_ratio": None,
                        "active_ratio": None,
                        "learning_rate": learning_rate,
                    }
                )
                writer.add_scalar(
                    "source_only/source_loss",
                    last_train_loss,
                    epoch,
                )
                writer.add_scalar("learning_rate", learning_rate, epoch)
                logger.info(
                    "held_out_subject=%s stage=source_only epoch=%d/%d loss=%.4f",
                    held_out_subject,
                    epoch + 1,
                    config.total_epochs,
                    last_train_loss,
                )
            target_test_loader = _classification_loader(
                fold.target_test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
            )
        else:
            if fold.target_adapt_dataset is None:
                raise RuntimeError("SPPM fold has no target-adaptation dataset.")
            if fold.source_signature_vectors is None:
                raise RuntimeError("SPPM fold has no source signature vectors.")
            n_target_adapt = len(fold.target_adapt_dataset)
            (
                source_loader,
                target_adapt_loader,
                target_test_loader,
                target_init_loader,
            ) = build_sppm_dataloaders(
                fold.source_dataset,
                fold.target_adapt_dataset,
                fold.target_test_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            for epoch in range(config.stage_i_epochs):
                last_train_loss = train_source_only_epoch(
                    source_loader,
                    model,
                    optimizer,
                    device,
                )
                _step_scheduler(scheduler)
                learning_rate = float(optimizer.param_groups[0]["lr"])
                epoch_rows.append(
                    {
                        "stage": "I",
                        "epoch": epoch,
                        "train_loss": last_train_loss,
                        "source_loss": last_train_loss,
                        "target_loss": None,
                        "accepted_ratio": None,
                        "active_ratio": None,
                        "learning_rate": learning_rate,
                    }
                )
                writer.add_scalar("stage_i/source_loss", last_train_loss, epoch)
                writer.add_scalar("learning_rate", learning_rate, epoch)
                logger.info(
                    "held_out_subject=%s stage=I epoch=%d/%d source_loss=%.4f",
                    held_out_subject,
                    epoch + 1,
                    config.stage_i_epochs,
                    last_train_loss,
                )

            source_eval_loader = _classification_loader(
                fold.source_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                generator=torch.Generator().manual_seed(fold_seed),
            )
            stage_i_result = evaluate_target(
                model,
                source_eval_loader,
                device,
                dataset_info.n_outputs,
            )
            stage_i_metrics = stage_i_result["metrics"]
            stage_i_source_eval_loss = float(stage_i_result["test_loss"])
            stage_i_source_eval_accuracy = float(stage_i_metrics["accuracy"])
            stage_i_source_eval_balanced_accuracy = float(
                stage_i_metrics["balanced_accuracy"]
            )
            epoch_rows[-1].update(
                {
                    "source_eval_loss": stage_i_source_eval_loss,
                    "source_eval_accuracy": stage_i_source_eval_accuracy,
                    "source_eval_balanced_accuracy": (
                        stage_i_source_eval_balanced_accuracy
                    ),
                }
            )
            writer.add_scalar(
                "stage_i/source_eval_loss",
                stage_i_source_eval_loss,
                config.stage_i_epochs - 1,
            )
            writer.add_scalar(
                "stage_i/source_eval_accuracy",
                stage_i_source_eval_accuracy,
                config.stage_i_epochs - 1,
            )
            writer.add_scalar(
                "stage_i/source_eval_balanced_accuracy",
                stage_i_source_eval_balanced_accuracy,
                config.stage_i_epochs - 1,
            )
            logger.info(
                "held_out_subject=%s stage=I complete source_eval_loss=%.4f "
                "source_eval_accuracy=%.3f source_eval_balanced_accuracy=%.3f "
                "source_confusion_matrix=%s",
                held_out_subject,
                stage_i_source_eval_loss,
                stage_i_source_eval_accuracy,
                stage_i_source_eval_balanced_accuracy,
                stage_i_metrics.get("confusion_matrix"),
            )

            prototypes, tolerances = build_shared_private_signature_prototypes(
                fold.source_signature_vectors,
                fold.source_labels,
                num_classes=dataset_info.n_outputs,
                floor=config.matching_tolerance_floor,
            )
            prototypes_t = torch.as_tensor(
                prototypes,
                dtype=torch.float32,
                device=device,
            )
            tolerances_t = torch.as_tensor(
                tolerances,
                dtype=torch.float32,
                device=device,
            )
            sppm_config = SPPMTrainingConfig(
                device=device,
                alpha=config.alpha,
                pseudo_threshold=config.pseudo_threshold,
            )
            initial_ratio, initial_stats = initialize_target_pseudo_labels(
                sppm_config,
                model,
                target_init_loader,
                prototypes_t,
                tolerances_t,
            )
            writer.add_scalar("stage_ii/initial_active_ratio", initial_ratio, 0)
            initial_signature_pass_count = initial_stats["signature_pass_count"]
            logger.info(
                "held_out_subject=%s initial_target_active_ratio=%.3f "
                "confidence_pass=%d signature_pass=%s accepted=%d",
                held_out_subject,
                initial_ratio,
                initial_stats["confidence_pass_count"],
                initial_signature_pass_count,
                initial_stats["accepted_count"],
            )

            for stage_epoch in range(config.stage_ii_epochs):
                stats = train_sppm_epoch(
                    sppm_config,
                    source_loader,
                    target_adapt_loader,
                    model,
                    optimizer,
                    prototypes_t,
                    tolerances_t,
                )
                _step_scheduler(scheduler)
                global_epoch = config.stage_i_epochs + stage_epoch
                learning_rate = float(optimizer.param_groups[0]["lr"])
                last_train_loss = float(stats["train_loss"])
                final_accepted_ratio = float(stats["accepted_ratio"])
                final_active_ratio = float(stats["active_ratio"])
                matching_stats = stats["matching_stats"]
                final_confidence_pass_ratio = float(
                    matching_stats["confidence_pass_ratio"]
                )
                signature_pass_ratio = matching_stats["signature_pass_ratio"]
                final_signature_pass_ratio = (
                    None if signature_pass_ratio is None else float(signature_pass_ratio)
                )
                epoch_rows.append(
                    {
                        "stage": "II",
                        "epoch": stage_epoch,
                        "train_loss": last_train_loss,
                        "source_loss": stats["source_loss"],
                        "target_loss": stats["target_loss"],
                        "confidence_pass_ratio": final_confidence_pass_ratio,
                        "signature_pass_ratio": final_signature_pass_ratio,
                        "accepted_ratio": final_accepted_ratio,
                        "active_ratio": final_active_ratio,
                        "learning_rate": learning_rate,
                    }
                )
                writer.add_scalar(
                    "stage_ii/train_loss",
                    last_train_loss,
                    stage_epoch,
                )
                writer.add_scalar(
                    "stage_ii/source_loss",
                    stats["source_loss"],
                    stage_epoch,
                )
                writer.add_scalar(
                    "stage_ii/target_loss",
                    stats["target_loss"],
                    stage_epoch,
                )
                writer.add_scalar(
                    "stage_ii/accepted_ratio",
                    final_accepted_ratio,
                    stage_epoch,
                )
                writer.add_scalar(
                    "stage_ii/active_ratio",
                    final_active_ratio,
                    stage_epoch,
                )
                writer.add_scalar("learning_rate", learning_rate, global_epoch)
                logger.info(
                    "held_out_subject=%s stage=II epoch=%d/%d loss=%.4f "
                    "source_loss=%.4f target_loss=%.4f "
                    "confidence_pass=%.3f signature_pass=%s "
                    "mean_refresh_acceptance=%.3f final_active=%.3f",
                    held_out_subject,
                    stage_epoch + 1,
                    config.stage_ii_epochs,
                    last_train_loss,
                    stats["source_loss"],
                    stats["target_loss"],
                    final_confidence_pass_ratio,
                    final_signature_pass_ratio,
                    final_accepted_ratio,
                    final_active_ratio,
                )

        result = evaluate_target(
            model,
            target_test_loader,
            device,
            dataset_info.n_outputs,
        )
        metrics = result["metrics"]
        test_metrics = _prefix_test_metrics(metrics)
        checkpoint_path = run_dirs["checkpoints"] / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        history_path = _write_history_csv(epoch_rows, run_dirs["history"])

        save_dataset_info(
            run_dir,
            dataset_info,
            extra={
                "held_out_subject": held_out_subject,
                "n_train_windows": len(fold.source_dataset),
                "n_target_adapt_windows": n_target_adapt,
                "n_test_windows": len(fold.target_test_dataset),
                "target_access": (
                    "none"
                    if config.protocol == SOURCE_ONLY
                    else "unlabeled_transductive"
                ),
                "canonicalization": config.canonicalization,
                "preprocessing": preprocessing_metadata,
                "config_fingerprint": fingerprint,
            },
        )
        final_metrics = {
            **{
                key: value
                for key, value in test_metrics.items()
                if key != "test_confusion_matrix"
            },
            "final_train_loss": last_train_loss,
            "stage_i_source_eval_loss": stage_i_source_eval_loss,
            "stage_i_source_eval_accuracy": stage_i_source_eval_accuracy,
            "stage_i_source_eval_balanced_accuracy": (
                stage_i_source_eval_balanced_accuracy
            ),
            "final_confidence_pass_ratio": final_confidence_pass_ratio,
            "final_signature_pass_ratio": final_signature_pass_ratio,
            "final_accepted_ratio": final_accepted_ratio,
            "final_active_ratio": final_active_ratio,
        }
        save_final_metrics(run_dir, final_metrics)

        run_id = (
            f"{held_out_label(held_out_subject)}__seed{config.seed}"
            f"__cfg{fingerprint}__{make_timestamp()}"
        )
        _save_run_metadata(
            run_dir,
            run_id=run_id,
            config=config,
            fingerprint=fingerprint,
            fold_seed=fold_seed,
            device=str(device),
            tensorboard_dir=tb_dir,
        )
        write_final_scalars(tb_dir, test_metrics)
        if "confusion_matrix" in metrics:
            write_confusion_matrix(
                tb_dir,
                metrics["confusion_matrix"],
                class_names=CLASS_NAMES_BY_CANONICALIZATION[
                    config.canonicalization
                ],
            )
        writer.flush()

        return _base_master_row(
            config=config,
            fingerprint=fingerprint,
            held_out_subject=held_out_subject,
            dataset_info=dataset_info,
            run_dir=run_dir,
            tb_dir=tb_dir,
            config_path=config_path,
            status="success",
            n_train_windows=len(fold.source_dataset),
            n_test_windows=len(fold.target_test_dataset),
            train_loss=last_train_loss,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            final_metrics_path=run_dirs["metrics"] / "final_metrics.yaml",
            dataset_info_path=run_dirs["metrics"] / "dataset_info.yaml",
            run_id=run_id,
            test_metrics=test_metrics,
        )
    finally:
        writer.close()


def _build_optimizer(
    config: CFSPMExperimentConfig,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    optimizer_class = torch.optim.Adam if config.optimizer == "adam" else torch.optim.AdamW
    return optimizer_class(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _build_scheduler(
    config: CFSPMExperimentConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.total_epochs - 1, 1),
    )


def _step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    if scheduler is not None:
        scheduler.step()


def _classification_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if len(dataset) == 0:
        raise ValueError("Cannot create a DataLoader for an empty dataset.")
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _base_master_row(
    *,
    config: CFSPMExperimentConfig,
    fingerprint: str,
    held_out_subject: int,
    dataset_info: Any,
    run_dir: Path,
    tb_dir: Path,
    config_path: Path,
    status: str,
    n_train_windows: int | None = None,
    n_test_windows: int | None = None,
    train_loss: float | None = None,
    checkpoint_path: Path | None = None,
    history_path: Path | None = None,
    final_metrics_path: Path | None = None,
    dataset_info_path: Path | None = None,
    run_id: str | None = None,
    test_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id
        or (
            f"{held_out_label(held_out_subject)}__seed{config.seed}"
            f"__cfg{fingerprint}__{make_timestamp()}"
        ),
        "run_dir": str(run_dir),
        "tensorboard_dir": str(tb_dir),
        "timestamp": make_timestamp(),
        "experiment": config.experiment_name,
        "dataset": DATASET_NAME,
        "model": config.model_name,
        "held_out_subject": held_out_subject,
        "seed": config.seed,
        "split_strategy": (
            "full_target_source_only"
            if config.protocol == SOURCE_ONLY
            else "full_target_transductive_sppm"
        ),
        "n_chans": dataset_info.n_chans,
        "n_outputs": dataset_info.n_outputs,
        "n_times": dataset_info.n_times,
        "sfreq": dataset_info.sfreq,
        "config_path": str(config_path),
        "status": status,
    }
    optional = {
        "n_train_windows": n_train_windows,
        "n_test_windows": n_test_windows,
        "train_loss": train_loss,
        "checkpoint_path": (
            None if checkpoint_path is None else str(checkpoint_path)
        ),
        "history_path": None if history_path is None else str(history_path),
        "final_metrics_path": (
            None if final_metrics_path is None else str(final_metrics_path)
        ),
        "dataset_info_path": (
            None if dataset_info_path is None else str(dataset_info_path)
        ),
    }
    row.update({key: value for key, value in optional.items() if value is not None})
    if test_metrics is not None:
        row.update(
            {
                key: value
                for key, value in test_metrics.items()
                if key != "test_confusion_matrix"
            }
        )
    return row


def current_result_rows(
    config: CFSPMExperimentConfig,
) -> list[dict[str, Any]]:
    repo_root = _find_repo_root(Path(__file__))
    output_root = _resolve_output_root(repo_root, config.output_root)
    path = _master_result_path(output_root, experiment=config.experiment_name)
    fingerprint = scientific_config_fingerprint(config)
    if not path.exists():
        return []
    token = f"__cfg{fingerprint}__"
    latest_by_subject: dict[int, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("model") != config.model_name
                or row.get("status") != "success"
                or token not in row.get("run_id", "")
                or not row.get("held_out_subject")
            ):
                continue
            latest_by_subject[int(row["held_out_subject"])] = (
                _coerce_result_row(row)
            )
    return [
        latest_by_subject[subject_id]
        for subject_id in sorted(latest_by_subject)
    ]


def _coerce_result_row(row: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(row)
    weight = coerced.get("n_test_windows")
    if weight not in (None, ""):
        try:
            coerced["n_test_windows"] = int(float(weight))
        except (TypeError, ValueError):
            pass
    for key, value in tuple(coerced.items()):
        if not key.startswith("test_") or value in (None, ""):
            continue
        try:
            coerced[key] = float(value)
        except (TypeError, ValueError):
            pass
    return coerced


def summarize_current_results(
    config: CFSPMExperimentConfig,
) -> dict[str, float]:
    return summarize_scalar_metrics(current_result_rows(config))


def _completed_subjects(
    path: Path,
    *,
    model_name: str,
    fingerprint: str,
) -> set[int]:
    if not path.exists():
        return set()
    token = f"__cfg{fingerprint}__"
    completed: set[int] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("model") == model_name
                and row.get("status") == "success"
                and token in row.get("run_id", "")
                and row.get("held_out_subject")
            ):
                completed.add(int(row["held_out_subject"]))
    return completed


def _master_result_path(output_root: Path, *, experiment: str) -> Path:
    safe_experiment = experiment.replace(" ", "_")
    return (
        output_root
        / "results"
        / DATASET_NAME
        / METHOD
        / f"results_master_{safe_experiment}.csv"
    )


def _prefix_test_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key if key.startswith("test_") else f"test_{key}": value
        for key, value in metrics.items()
    }


def _write_history_csv(
    rows: list[dict[str, Any]],
    history_dir: Path,
) -> Path:
    path = history_dir / "history.csv"
    fieldnames = [
        "stage",
        "epoch",
        "train_loss",
        "source_loss",
        "source_eval_loss",
        "source_eval_accuracy",
        "source_eval_balanced_accuracy",
        "target_loss",
        "confidence_pass_ratio",
        "signature_pass_ratio",
        "accepted_ratio",
        "active_ratio",
        "learning_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_run_metadata(
    run_dir: Path,
    *,
    run_id: str,
    config: CFSPMExperimentConfig,
    fingerprint: str,
    fold_seed: int,
    device: str,
    tensorboard_dir: Path,
) -> None:
    metadata = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "experiment": config.experiment_name,
        "model": config.model_name,
        "protocol": config.protocol,
        "canonicalization": config.canonicalization,
        "preprocessing": config.preprocessing,
        "config_fingerprint": fingerprint,
        "seed": config.seed,
        "fold_seed": fold_seed,
        "device": device,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "braindecode_version": getattr(braindecode, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git_output(["git", "status", "--porcelain"])),
        "status": "success",
    }
    path = run_dir / "metrics" / "run_metadata.yaml"
    OmegaConf.save(OmegaConf.create(_jsonable(metadata)), path)


def _participants_path(
    repo_root: Path,
    figshare_cache_dir: Path | None,
    *,
    prefer_figshare: bool = False,
) -> Path:
    project_path = (
        repo_root
        / "data"
        / "moabb"
        / "MNE-liu2024-data"
        / "files"
        / "participants.tsv"
    )
    cache_dir = repo_root / "data" / "figshare" / "liu2024"
    if figshare_cache_dir is not None:
        cache_dir = Path(figshare_cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = repo_root / cache_dir
    figshare_path = cache_dir / "participants.tsv"
    candidates = (
        (figshare_path, project_path)
        if prefer_figshare
        else (project_path, figshare_path)
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Liu2024 participants.tsv is required for affected/unaffected "
        f"canonicalization. Expected {project_path} or {figshare_path}."
    )


def _resolve_output_root(repo_root: Path, configured: Path | None) -> Path:
    if configured is None:
        return repo_root / "outputs"
    configured = Path(configured)
    return configured if configured.is_absolute() else repo_root / configured


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            args,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
