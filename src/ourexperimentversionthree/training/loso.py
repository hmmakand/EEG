"""Train one or all LOSO folds on the fixed broadcast-11 dataset."""

from __future__ import annotations

import copy
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm, trange

from src.datautils.PlvLiu2024Broadcast11 import PlvLiu2024GraphDataset
from src.ourexperimentversionthree.data import (
    GraphDataLoaderConfig,
    LosoDataLoaderBundle,
    create_loso_dataloaders,
    load_dataset,
)
from src.ourexperimentversionthree.model import EEGGCN1, EEGGCN1Config

from .config import TrainingConfig
from .engine import evaluate, resolve_device, set_seed, train_epoch
from .metrics import ClassificationMetrics


DATASET_VARIANT = "manuscript_broadcast_11_v3"
EXPECTED_GRAPH_FORMAT_VERSION = 3
EXPECTED_NODES = 29
EXPECTED_NODE_FEATURES = 11
EXPECTED_GRAPH_FEATURES = 0
EXPECTED_FEATURE_NAMES = (
    "MAV",
    "STD",
    "PSD",
    "betweenness",
    "PersistenceEntropy_0",
    "landscape1_0",
    "landscape1_1",
    "landscape2_0",
    "landscape2_1",
    "betti_0",
    "betti_1",
)


@dataclass(frozen=True)
class EpochRecord:
    """Training and validation metrics captured after one epoch."""

    epoch: int
    training: ClassificationMetrics
    validation: ClassificationMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "training": self.training.as_dict(),
            "validation": self.validation.as_dict(),
        }


@dataclass(frozen=True)
class FoldResult:
    """Best-checkpoint test result and complete history for one LOSO fold."""

    test_subject_id: int
    validation_subject_ids: tuple[int, ...]
    initialization_seed: int
    best_epoch: int
    epochs_ran: int
    test: ClassificationMetrics
    history: tuple[EpochRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "test_subject_id": self.test_subject_id,
            "validation_subject_ids": list(self.validation_subject_ids),
            "initialization_seed": self.initialization_seed,
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "test": self.test.as_dict(),
            "history": [record.as_dict() for record in self.history],
        }


def _log(message: str, *, enabled: bool) -> None:
    """Write progress-safe human-readable details to stderr."""

    if enabled:
        tqdm.write(message, file=sys.stderr)


def _subject_ids(dataset: PlvLiu2024GraphDataset) -> tuple[int, ...]:
    """Return sorted IDs represented by the data instead of assuming 1--50."""

    subjects = tuple(sorted(map(int, np.unique(dataset.arrays.subject_ids))))
    if len(subjects) < 3:
        raise ValueError(
            "The dataset must contain at least three subjects for "
            "train/validation/test splitting"
        )
    return subjects


def _validate_dataset_schema(dataset: PlvLiu2024GraphDataset) -> None:
    """Enforce the experiment's fixed broadcast-11 input contract."""

    metadata = dataset.metadata
    expected_metadata: dict[str, object] = {
        "graph_format_version": EXPECTED_GRAPH_FORMAT_VERSION,
        "dataset_variant": DATASET_VARIANT,
        "n_nodes": EXPECTED_NODES,
        "n_node_features": EXPECTED_NODE_FEATURES,
        "n_graph_features": EXPECTED_GRAPH_FEATURES,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if tuple(metadata.get("node_feature_names", ())) != EXPECTED_FEATURE_NAMES:
        mismatches["node_feature_names"] = (
            metadata.get("node_feature_names"),
            list(EXPECTED_FEATURE_NAMES),
        )
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Unexpected broadcast-11 dataset schema: {details}")

    graph_count = len(dataset)
    expected_node_shape = (graph_count, EXPECTED_NODES, EXPECTED_NODE_FEATURES)
    if tuple(dataset.arrays.node_features.shape) != expected_node_shape:
        raise ValueError(
            "node_features shape does not match the broadcast-11 contract: "
            f"{tuple(dataset.arrays.node_features.shape)} != {expected_node_shape}"
        )
    if metadata.get("n_graphs") != graph_count:
        raise ValueError(
            f"metadata n_graphs={metadata.get('n_graphs')!r} does not match "
            f"the loaded graph count {graph_count}"
        )
    if dataset.arrays.graph_features is not None:
        raise ValueError(
            "Version three expects no separate graph features; all 11 values "
            "must be present in graph.x"
        )


def _initialization_seed(config: TrainingConfig, test_subject_id: int) -> int:
    """Resolve the documented model/dropout seed policy for one fold."""

    seed = (
        config.seed + test_subject_id
        if config.seed_strategy == "per_fold"
        else config.seed
    )
    if seed > 2**32 - 1:
        raise ValueError(
            "base seed + test_subject_id exceeds the supported 32-bit seed range"
        )
    return seed


def _default_run_name() -> str:
    """Return a collision-resistant, sortable UTC run name."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _prepare_run_directory(config: TrainingConfig) -> Path:
    """Create one isolated run directory and enforce overwrite protection."""

    run_dir = config.output_dir / (config.run_name or _default_run_name())
    if run_dir.exists() and any(run_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Choose another --run-name "
            "or pass --overwrite explicitly."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _serializable_config(config: TrainingConfig) -> dict[str, Any]:
    values = asdict(config)
    values["output_dir"] = str(config.output_dir)
    return values


def _git_provenance() -> dict[str, Any]:
    """Return best-effort repository revision information."""

    repository_root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                cwd=repository_root,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    """Return core package versions without failing on missing metadata."""

    versions: dict[str, str | None] = {}
    for package in ("numpy", "torch", "torch-geometric", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(unique, counts, strict=True)
    }


def _dataset_provenance(
    dataset: PlvLiu2024GraphDataset,
    subjects: tuple[int, ...],
) -> dict[str, Any]:
    """Build dataset identity, schema, and observed distribution metadata."""

    return {
        "path": str(dataset.data_dir),
        "metadata": dataset.metadata,
        "observed": {
            "graphs": len(dataset),
            "node_features_shape": list(dataset.arrays.node_features.shape),
            "subjects": list(subjects),
            "graphs_per_subject": _counts(
                np.asarray(dataset.arrays.subject_ids)
            ),
            "class_counts": _counts(np.asarray(dataset.arrays.labels)),
        },
    }


def _run_manifest(
    *,
    config: TrainingConfig,
    dataset: PlvLiu2024GraphDataset,
    model_config: EEGGCN1Config,
    resolved_device: torch.device,
    subjects: tuple[int, ...],
    requested_test_subjects: tuple[int, ...],
    run_dir: Path,
) -> dict[str, Any]:
    """Build the provenance shared by all folds in one invocation."""

    return {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_dir.name,
        "run_directory": str(run_dir),
        "experiment": "ourexperimentversionthree",
        "dataset_variant": DATASET_VARIANT,
        "requested_test_subjects": list(requested_test_subjects),
        "training_config": _serializable_config(config),
        "resolved_device": str(resolved_device),
        "model_config": asdict(model_config),
        "dataset": _dataset_provenance(dataset, subjects),
        "seed_policy": {
            "strategy": config.seed_strategy,
            "base_seed": config.seed,
            "validation_split_and_loader_seed": config.seed,
            "initialization_description": (
                "Every fold uses the base seed."
                if config.seed_strategy == "shared"
                else "Each fold uses base_seed + test_subject_id."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "command": sys.argv,
        },
        "git": _git_provenance(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON artifact after successful serialization."""

    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    _write_json(run_dir / "run_manifest.json", manifest)


def _complete_manifest(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    summary: dict[str, Any],
) -> None:
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"] = summary
    _write_manifest(run_dir, manifest)


def _log_run_context(
    *,
    config: TrainingConfig,
    dataset: PlvLiu2024GraphDataset,
    device: torch.device,
    subjects: tuple[int, ...],
    run_dir: Path | None,
    enabled: bool,
) -> None:
    """Print enough context to interpret a captured training log."""

    metadata = dataset.metadata
    _log("", enabled=enabled)
    _log("=== Liu2024 broadcast-11 GCN training run ===", enabled=enabled)
    _log(f"Dataset variant : {DATASET_VARIANT}", enabled=enabled)
    _log(f"Dataset path    : {dataset.data_dir}", enabled=enabled)
    _log(
        f"Dataset shape   : {len(dataset):,} graphs, "
        f"{metadata['n_nodes']} nodes, "
        f"{metadata['n_node_features']} node features",
        enabled=enabled,
    )
    _log(
        "Feature columns : " + ", ".join(EXPECTED_FEATURE_NAMES),
        enabled=enabled,
    )
    _log(
        f"Subjects        : {len(subjects)} IDs from data; "
        f"{list(subjects)}",
        enabled=enabled,
    )
    _log(
        "Class counts    : " + str(_counts(np.asarray(dataset.arrays.labels))),
        enabled=enabled,
    )
    _log(f"Device          : {device}", enabled=enabled)
    _log(
        f"Training        : epochs={config.epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate:g}, weight_decay={config.weight_decay:g}, "
        f"gradient_clip={config.gradient_clip_norm}",
        enabled=enabled,
    )
    _log(
        f"Early stopping  : patience={config.patience}, "
        f"minimum improvement={config.minimum_improvement:g}",
        enabled=enabled,
    )
    _log(
        f"Seed policy     : {config.seed_strategy} (base seed={config.seed})",
        enabled=enabled,
    )
    _log(
        f"Outputs         : {run_dir if run_dir is not None else 'disabled'}",
        enabled=enabled,
    )


def _split_class_counts(
    dataset: PlvLiu2024GraphDataset,
    graph_indices: np.ndarray,
) -> dict[str, int]:
    return _counts(np.asarray(dataset.arrays.labels[graph_indices]))


def _validate_normalization(loaders: LosoDataLoaderBundle) -> None:
    mean = loaders.normalization.mean
    standard_deviation = loaders.normalization.standard_deviation
    if tuple(mean.shape) != (EXPECTED_NODE_FEATURES,) or tuple(
        standard_deviation.shape
    ) != (EXPECTED_NODE_FEATURES,):
        raise ValueError(
            "Feature normalization must contain 11 means and 11 standard "
            "deviations"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(
        standard_deviation
    ).all():
        raise ValueError("Feature normalization contains non-finite values")
    if torch.any(standard_deviation <= 0):
        raise ValueError("Feature standard deviations must be positive")


def _save_fold_outputs(
    result: FoldResult,
    model: nn.Module,
    config: TrainingConfig,
    loaders: LosoDataLoaderBundle,
    model_config: EEGGCN1Config,
    resolved_device: torch.device,
    dataset: PlvLiu2024GraphDataset,
    run_dir: Path,
) -> None:
    """Persist one selected model with inference and provenance information."""

    fold_dir = run_dir / f"subject_{result.test_subject_id:02d}"
    if fold_dir.exists() and any(fold_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Fold output directory is not empty: {fold_dir}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    normalization = {
        "mean": loaders.normalization.mean.detach().cpu(),
        "standard_deviation": (
            loaders.normalization.standard_deviation.detach().cpu()
        ),
    }
    split = {
        "train_subject_ids": list(loaders.split.train_subject_ids),
        "validation_subject_ids": list(
            loaders.split.validation_subject_ids
        ),
        "test_subject_id": loaders.split.test_subject_id,
        "train_graph_indices": loaders.split.train_graph_indices.tolist(),
        "validation_graph_indices": (
            loaders.split.validation_graph_indices.tolist()
        ),
        "test_graph_indices": loaders.split.test_graph_indices.tolist(),
    }
    checkpoint = {
        "checkpoint_type": "best_validation_loss",
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": _serializable_config(config),
        "resolved_device": str(resolved_device),
        "dataset_path": str(dataset.data_dir),
        "dataset_metadata": dataset.metadata,
        "node_feature_names": list(EXPECTED_FEATURE_NAMES),
        "feature_normalization": normalization,
        "split": split,
        "initialization_seed": result.initialization_seed,
        "result": result.as_dict(),
    }
    checkpoint_path = fold_dir / "best_model.pt"
    temporary_checkpoint = fold_dir / ".best_model.pt.tmp"
    torch.save(checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)

    _write_json(
        fold_dir / "result.json",
        {
            "result": result.as_dict(),
            "training_config": _serializable_config(config),
            "resolved_device": str(resolved_device),
            "model_config": asdict(model_config),
            "dataset_path": str(dataset.data_dir),
            "node_feature_names": list(EXPECTED_FEATURE_NAMES),
            "normalization": {
                "mean": normalization["mean"].tolist(),
                "standard_deviation": normalization[
                    "standard_deviation"
                ].tolist(),
            },
            "split": split,
        },
    )


def train_loso_fold(
    test_subject_id: int,
    config: TrainingConfig | None = None,
    *,
    dataset: PlvLiu2024GraphDataset | None = None,
    show_progress: bool = True,
    _run_dir: Path | None = None,
    _dataset_validated: bool = False,
) -> FoldResult:
    """Train, select on validation, and test one held-out subject."""

    resolved_config = config or TrainingConfig()
    device = resolve_device(resolved_config.device)
    dataset_loaded_here = dataset is None
    graph_dataset = dataset if dataset is not None else load_dataset()
    if not _dataset_validated:
        _validate_dataset_schema(graph_dataset)
    available_subjects = _subject_ids(graph_dataset)
    if test_subject_id not in available_subjects:
        raise ValueError(
            f"Unknown test subject {test_subject_id}; available subjects: "
            f"{available_subjects}"
        )

    initialization_seed = _initialization_seed(
        resolved_config, test_subject_id
    )
    set_seed(initialization_seed)
    loader_config = GraphDataLoaderConfig(
        batch_size=resolved_config.batch_size,
        validation_subjects=resolved_config.validation_subjects,
        num_workers=resolved_config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=resolved_config.num_workers > 0,
        seed=resolved_config.seed,
    )
    loaders = create_loso_dataloaders(
        test_subject_id=test_subject_id,
        config=loader_config,
        dataset=graph_dataset,
        dataset_validated=dataset_loaded_here or _dataset_validated,
    )
    _validate_normalization(loaders)
    model_config = EEGGCN1Config(input_features=EXPECTED_NODE_FEATURES, nodes=EXPECTED_NODES)

    run_dir = _run_dir
    manifest: dict[str, Any] | None = None
    if resolved_config.save_outputs and run_dir is None:
        run_dir = _prepare_run_directory(resolved_config)
        manifest = _run_manifest(
            config=resolved_config,
            dataset=graph_dataset,
            model_config=model_config,
            resolved_device=device,
            subjects=available_subjects,
            requested_test_subjects=(test_subject_id,),
            run_dir=run_dir,
        )
        _write_manifest(run_dir, manifest)

    model = EEGGCN1(model_config).to(device)
    if not _dataset_validated:
        _log_run_context(
            config=resolved_config,
            dataset=graph_dataset,
            device=device,
            subjects=available_subjects,
            run_dir=run_dir,
            enabled=show_progress,
        )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    split = loaders.split
    _log("", enabled=show_progress)
    _log(f"--- Fold: test subject {test_subject_id} ---", enabled=show_progress)
    _log(
        f"Split graphs    : train={len(split.train_graph_indices):,}, "
        f"validation={len(split.validation_graph_indices):,}, "
        f"test={len(split.test_graph_indices):,}",
        enabled=show_progress,
    )
    _log(
        "Split subjects  : "
        f"train={len(split.train_subject_ids)}, "
        f"validation={list(split.validation_subject_ids)}, "
        f"test={split.test_subject_id}",
        enabled=show_progress,
    )
    _log(
        "Split classes   : "
        f"train={_split_class_counts(graph_dataset, split.train_graph_indices)}, "
        "validation="
        f"{_split_class_counts(graph_dataset, split.validation_graph_indices)}, "
        f"test={_split_class_counts(graph_dataset, split.test_graph_indices)}",
        enabled=show_progress,
    )
    _log(
        f"Normalization   : fitted on training nodes only; "
        f"std range=[{loaders.normalization.standard_deviation.min().item():.3g}, "
        f"{loaders.normalization.standard_deviation.max().item():.3g}]",
        enabled=show_progress,
    )
    _log(
        f"Initialization  : seed={initialization_seed}, "
        f"trainable parameters={trainable_parameters:,}",
        enabled=show_progress,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )
    loss_function = nn.NLLLoss()
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[EpochRecord] = []
    fold_started = time.perf_counter()

    epochs = trange(
        1,
        resolved_config.epochs + 1,
        desc=f"Subject {test_subject_id:02d}",
        unit="epoch",
        disable=not show_progress,
        leave=False,
    )
    for epoch in epochs:
        epoch_started = time.perf_counter()
        training_metrics = train_epoch(
            model,
            loaders.train,
            loss_function,
            optimizer,
            device,
            gradient_clip_norm=resolved_config.gradient_clip_norm,
        )
        validation_metrics = evaluate(
            model, loaders.validation, loss_function, device
        )
        history.append(
            EpochRecord(epoch, training_metrics, validation_metrics)
        )
        improved = (
            validation_metrics.loss
            < best_validation_loss - resolved_config.minimum_improvement
        )
        if improved:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            checkpoint_status = "new best"
        else:
            epochs_without_improvement += 1
            checkpoint_status = (
                f"patience {epochs_without_improvement}/"
                f"{resolved_config.patience}"
            )
        epoch_seconds = time.perf_counter() - epoch_started
        _log(
            f"Epoch {epoch:03d}/{resolved_config.epochs:03d} | "
            f"train loss={training_metrics.loss:.4f} "
            f"acc={training_metrics.accuracy:.3f} "
            f"f1={training_metrics.f1:.3f} "
            f"auc={training_metrics.auc:.3f} | "
            f"val loss={validation_metrics.loss:.4f} "
            f"acc={validation_metrics.accuracy:.3f} "
            f"f1={validation_metrics.f1:.3f} "
            f"auc={validation_metrics.auc:.3f} | "
            f"best={best_validation_loss:.4f}@{best_epoch} | "
            f"{checkpoint_status} | {epoch_seconds:.1f}s",
            enabled=show_progress,
        )
        epochs.set_postfix(
            tr_loss=f"{training_metrics.loss:.4f}",
            val_loss=f"{validation_metrics.loss:.4f}",
            val_auc=f"{validation_metrics.auc:.3f}",
            best=f"{best_validation_loss:.4f}@{best_epoch}",
            patience=(
                f"{epochs_without_improvement}/{resolved_config.patience}"
            ),
        )
        if epochs_without_improvement >= resolved_config.patience:
            _log(
                f"Early stopping  : no validation-loss improvement for "
                f"{resolved_config.patience} epochs",
                enabled=show_progress,
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders.test, loss_function, device)
    result = FoldResult(
        test_subject_id=test_subject_id,
        validation_subject_ids=split.validation_subject_ids,
        initialization_seed=initialization_seed,
        best_epoch=best_epoch,
        epochs_ran=len(history),
        test=test_metrics,
        history=tuple(history),
    )
    _log(
        f"Fold complete   : best_epoch={best_epoch}, epochs_ran={len(history)}, "
        f"test_loss={test_metrics.loss:.4f}, "
        f"test_acc={test_metrics.accuracy:.3f}, "
        f"test_f1={test_metrics.f1:.3f}, "
        f"test_recall={test_metrics.recall:.3f}, "
        f"test_precision={test_metrics.precision:.3f}, "
        f"test_auc={test_metrics.auc:.3f}, "
        f"elapsed={time.perf_counter() - fold_started:.1f}s",
        enabled=show_progress,
    )

    if resolved_config.save_outputs:
        if run_dir is None:
            raise RuntimeError("Output saving requires a prepared run directory")
        _save_fold_outputs(
            result,
            model,
            resolved_config,
            loaders,
            model_config,
            device,
            graph_dataset,
            run_dir,
        )
        if manifest is not None:
            _complete_manifest(
                run_dir,
                manifest,
                summary={"folds": 1, **result.test.as_dict()},
            )
            _log(f"Results saved   : {run_dir}", enabled=show_progress)
    return result


def summarize_results(results: list[FoldResult]) -> dict[str, Any]:
    """Summarize subject-level metrics with equal subject weighting."""

    if not results:
        raise ValueError("At least one fold result is required")
    summary: dict[str, Any] = {"folds": len(results)}
    for metric_name in ("loss", "accuracy", "f1", "recall", "precision", "auc"):
        values = np.asarray(
            [getattr(result.test, metric_name) for result in results],
            dtype=np.float64,
        )
        summary[f"mean_{metric_name}"] = float(values.mean())
        summary[f"std_{metric_name}"] = float(values.std())
    summary["total_test_examples"] = sum(
        result.test.examples for result in results
    )
    return summary


def _save_loso_summary(
    results: list[FoldResult],
    summary: dict[str, Any],
    config: TrainingConfig,
    run_dir: Path,
) -> None:
    _write_json(
        run_dir / "loso_results.json",
        {
            "training_config": _serializable_config(config),
            "summary": summary,
            "folds": [result.as_dict() for result in results],
        },
    )
    csv_path = run_dir / "loso_results.csv"
    temporary_csv = run_dir / ".loso_results.csv.tmp"
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "test_subject_id",
            "initialization_seed",
            "best_epoch",
            "epochs_ran",
            "loss",
            "accuracy",
            "f1",
            "recall",
            "precision",
            "auc",
            "examples",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "test_subject_id": result.test_subject_id,
                    "initialization_seed": result.initialization_seed,
                    "best_epoch": result.best_epoch,
                    "epochs_ran": result.epochs_ran,
                    **result.test.as_dict(),
                }
            )
    temporary_csv.replace(csv_path)


def train_all_loso_folds(
    config: TrainingConfig | None = None,
    *,
    show_progress: bool = True,
) -> tuple[list[FoldResult], dict[str, Any]]:
    """Train one fold for every subject represented by the dataset."""

    resolved_config = config or TrainingConfig()
    dataset = load_dataset()
    _validate_dataset_schema(dataset)
    subject_ids = _subject_ids(dataset)
    if resolved_config.validation_subjects > len(subject_ids) - 2:
        raise ValueError(
            "validation_subjects must leave one test subject and at least "
            "one training subject"
        )
    device = resolve_device(resolved_config.device)
    model_config = EEGGCN1Config(input_features=EXPECTED_NODE_FEATURES, nodes=EXPECTED_NODES)
    run_dir = (
        _prepare_run_directory(resolved_config)
        if resolved_config.save_outputs
        else None
    )
    manifest: dict[str, Any] | None = None
    if run_dir is not None:
        manifest = _run_manifest(
            config=resolved_config,
            dataset=dataset,
            model_config=model_config,
            resolved_device=device,
            subjects=subject_ids,
            requested_test_subjects=subject_ids,
            run_dir=run_dir,
        )
        _write_manifest(run_dir, manifest)
    _log_run_context(
        config=resolved_config,
        dataset=dataset,
        device=device,
        subjects=subject_ids,
        run_dir=run_dir,
        enabled=show_progress,
    )

    results: list[FoldResult] = []
    subjects = tqdm(
        subject_ids,
        desc="LOSO folds",
        unit="subject",
        disable=not show_progress,
    )
    for subject_id in subjects:
        result = train_loso_fold(
            subject_id,
            resolved_config,
            dataset=dataset,
            show_progress=show_progress,
            _run_dir=run_dir,
            _dataset_validated=True,
        )
        results.append(result)
        subjects.set_postfix(
            test_auc=f"{result.test.auc:.3f}",
            mean_auc=(
                f"{np.mean([item.test.auc for item in results]):.3f}"
            ),
        )

    summary = summarize_results(results)
    if run_dir is not None:
        _save_loso_summary(results, summary, resolved_config, run_dir)
        if manifest is None:
            raise RuntimeError("A saved run requires a run manifest")
        _complete_manifest(run_dir, manifest, summary=summary)
    _log("", enabled=show_progress)
    _log("=== LOSO run complete ===", enabled=show_progress)
    _log(
        f"Folds={summary['folds']}, "
        f"accuracy={summary['mean_accuracy']:.3f} +/- "
        f"{summary['std_accuracy']:.3f}, "
        f"f1={summary['mean_f1']:.3f} +/- "
        f"{summary['std_f1']:.3f}, "
        f"recall={summary['mean_recall']:.3f} +/- "
        f"{summary['std_recall']:.3f}, "
        f"precision={summary['mean_precision']:.3f} +/- "
        f"{summary['std_precision']:.3f}, "
        f"auc={summary['mean_auc']:.3f} +/- "
        f"{summary['std_auc']:.3f}",
        enabled=show_progress,
    )
    if run_dir is not None:
        _log(f"Results saved   : {run_dir}", enabled=show_progress)
    return results, summary
