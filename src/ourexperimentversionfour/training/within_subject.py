"""Within-subject classification: a diagnostic sanity check alongside LOSO.

Trains and evaluates on one subject's own trials only, with no cross-subject
generalization involved -- see ``WITHIN_SUBJECT_PLAN.md`` for the full
rationale (isolating "can the model fit this data at all" and "is there
decodable signal at all" from the harder cross-subject LOSO problem).

Mirrors ``hyperparameter_search.py``'s reuse of ``engine.py``'s low-level
training/evaluation building blocks and ``EEGGCN1`` exactly as
``loso.py`` does; only fold construction, config, and orchestration differ
here, since the two evaluation protocols do not share a shape (no early
stopping, no validation split, one subject's trials per fold rather than a
cross-subject group). Generic (non-LOSO-specific) provenance/logging/run-
directory helpers are imported from ``.loso`` rather than duplicated.
"""

from __future__ import annotations

import copy
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm, trange

from src.datautils.graphdataversionone.saved_dataset import SavedDataset
from src.ourexperimentversionfour.data import (
    EXPECTED_NODES,
    EXPECTED_NODE_FEATURES,
    GraphDataLoaderConfig,
)
from src.ourexperimentversionfour.data.combinations import Combination, get_combination
from src.ourexperimentversionfour.data.within_subject_split import (
    WithinSubjectFold,
    create_within_subject_folds,
)
from src.ourexperimentversionfour.model import EEGGCN1, EEGGCN1Config

from .config import WithinSubjectConfig
from .engine import build_optimizer, evaluate, resolve_device, set_seed, train_epoch
from .loso import (
    _bars_enabled,
    _default_run_name,
    _git_provenance,
    _log,
    _package_versions,
    _prepare_run_directory,
    _serializable_config,
    _write_json,
)
from .metrics import ClassificationMetrics


@dataclass(frozen=True)
class WithinSubjectEpochRecord:
    """Training metrics captured after one epoch (no validation split here)."""

    epoch: int
    training: ClassificationMetrics

    def as_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "training": self.training.as_dict()}


@dataclass(frozen=True)
class WithinSubjectFoldResult:
    """Best-checkpoint evaluation result and history for one within-subject fold."""

    subject_id: int
    fold: int
    repeat: int
    best_epoch: int
    epochs_ran: int
    evaluation: ClassificationMetrics
    history: tuple[WithinSubjectEpochRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "fold": self.fold,
            "repeat": self.repeat,
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "evaluation": self.evaluation.as_dict(),
            "history": [record.as_dict() for record in self.history],
        }


def _dataset_subject_ids(dataset: SavedDataset) -> np.ndarray:
    return np.asarray([sample["subject"] for sample in dataset.samples], dtype=np.int64)


def train_within_subject_fold(
    subject_id: int,
    fold: WithinSubjectFold,
    *,
    combination: Combination,
    dataset: SavedDataset,
    config: WithinSubjectConfig,
    repeat: int,
    show_progress: bool = True,
) -> WithinSubjectFoldResult:
    """Train on one within-subject fold's training split, evaluate once.

    No early stopping: trains for the full ``config.epochs`` budget and
    checkpoints on best **training** loss (zero-leakage, since it never
    touches the evaluation fold) instead of the validation-loss criterion
    ``train_loso_fold`` uses -- there is no validation split to select on
    here (see ``WITHIN_SUBJECT_PLAN.md``).

    Normalization is fit strictly from this fold's ``train_graph_indices``,
    never from its evaluation trials -- a stricter scope than the LOSO
    pipeline's (which fits from a subject's whole trial set; see
    ``AUDIT.md`` item 3). This is one subject's ~32 trials, so refitting it
    fresh per fold is cheap; there is nothing shareable across folds since
    each fold's training set differs.
    """

    device = resolve_device(config.device)
    loader_config = GraphDataLoaderConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        seed=config.seed,
    )
    fold_normalization = combination.fit_feature_normalization(
        dataset,
        epsilon=loader_config.normalization_epsilon,
        graph_indices=fold.train_graph_indices,
    )
    train_loader, evaluation_loader, _normalization = (
        combination.create_within_subject_dataloaders(
            dataset=dataset,
            train_graph_indices=fold.train_graph_indices,
            evaluation_graph_indices=fold.evaluation_graph_indices,
            config=loader_config,
            dataset_validated=True,
            normalization=fold_normalization,
        )
    )

    set_seed(config.seed)
    model = EEGGCN1(
        EEGGCN1Config(input_features=EXPECTED_NODE_FEATURES, nodes=EXPECTED_NODES)
    ).to(device)
    optimizer = build_optimizer(
        model,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )
    loss_function = nn.NLLLoss()

    best_training_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[WithinSubjectEpochRecord] = []

    epochs = trange(
        1,
        config.epochs + 1,
        desc=f"Subject {subject_id:02d} fold {fold.fold} repeat {repeat}",
        unit="epoch",
        disable=not _bars_enabled(show_progress),
        leave=False,
    )
    for epoch in epochs:
        training_metrics = train_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        history.append(WithinSubjectEpochRecord(epoch, training_metrics))
        if training_metrics.loss < best_training_loss:
            best_training_loss = training_metrics.loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epochs.set_postfix(
            tr_loss=f"{training_metrics.loss:.4f}",
            best=f"{best_training_loss:.4f}@{best_epoch}",
        )
        _log(
            f"Subject {subject_id:02d} fold {fold.fold} repeat {repeat} | "
            f"epoch {epoch:03d}/{config.epochs:03d} | "
            f"train loss={training_metrics.loss:.4f} "
            f"acc={training_metrics.accuracy:.3f} | "
            f"best={best_training_loss:.4f}@{best_epoch}",
            enabled=show_progress,
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    evaluation_metrics = evaluate(model, evaluation_loader, loss_function, device)
    return WithinSubjectFoldResult(
        subject_id=subject_id,
        fold=fold.fold,
        repeat=repeat,
        best_epoch=best_epoch,
        epochs_ran=len(history),
        evaluation=evaluation_metrics,
        history=tuple(history),
    )


def train_within_subject(
    subject_id: int,
    config: WithinSubjectConfig,
    *,
    dataset: SavedDataset,
    combination: Combination,
    show_progress: bool = True,
) -> tuple[WithinSubjectFoldResult, ...]:
    """Run every ``(fold, repeat)`` pair for one subject.

    Each repeat rebuilds the ``k``-fold split with a different seed
    (``config.seed + repeat``) so ``repeats`` gives independent estimates
    over different fold membership, not just different model
    initializations on the same split. Each fold fits its own train-only
    normalization (see :func:`train_within_subject_fold`).
    """

    subject_ids = _dataset_subject_ids(dataset)
    labels = np.asarray(dataset.labels)

    results: list[WithinSubjectFoldResult] = []
    for repeat in range(config.repeats):
        folds = create_within_subject_folds(
            subject_ids,
            labels,
            subject_id,
            k=config.folds,
            seed=config.seed + repeat,
        )
        for fold in folds:
            results.append(
                train_within_subject_fold(
                    subject_id,
                    fold,
                    combination=combination,
                    dataset=dataset,
                    config=config,
                    repeat=repeat,
                    show_progress=show_progress,
                )
            )
    return tuple(results)


def summarize_within_subject_results(
    results: list[WithinSubjectFoldResult],
    *,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize per-subject-averaged metrics with equal subject weighting.

    Two-level aggregation, not a generalization of ``loso.summarize_results``
    (which has one result per subject already): every ``(fold, repeat)``
    result for a subject is first averaged into one per-subject mean -- so a
    subject run with more repeats doesn't get more weight -- then a
    percentile-bootstrap confidence interval is computed *across subjects* on
    that per-subject-mean list, exactly mirroring ``summarize_results``'s
    "equal subject weighting" philosophy for the across-subject step.
    """

    if not results:
        raise ValueError("At least one fold result is required")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in the interval (0, 1)")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")

    metric_names = (
        "loss",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "recall",
        "precision",
        "auc",
        "cohens_kappa",
    )

    subject_ids = sorted({result.subject_id for result in results})
    per_subject_means: dict[str, list[float]] = {metric: [] for metric in metric_names}
    per_subject_summary: dict[str, dict[str, float]] = {}
    for subject_id in subject_ids:
        subject_results = [
            result for result in results if result.subject_id == subject_id
        ]
        subject_means: dict[str, float] = {}
        for metric_name in metric_names:
            values = np.asarray(
                [
                    getattr(result.evaluation, metric_name)
                    for result in subject_results
                ],
                dtype=np.float64,
            )
            mean_value = float(values.mean())
            per_subject_means[metric_name].append(mean_value)
            subject_means[metric_name] = mean_value
        per_subject_summary[str(subject_id)] = subject_means

    summary: dict[str, Any] = {
        "subjects": len(subject_ids),
        "total_folds": len(results),
        "per_subject": per_subject_summary,
    }
    generator = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    for metric_name in metric_names:
        values = np.asarray(per_subject_means[metric_name], dtype=np.float64)
        summary[f"mean_{metric_name}"] = float(values.mean())
        summary[f"std_{metric_name}"] = float(values.std())

        resampled_indices = generator.integers(
            0, len(values), size=(bootstrap_resamples, len(values))
        )
        resampled_means = values[resampled_indices].mean(axis=1)
        low, high = np.quantile(resampled_means, (tail, 1 - tail))
        summary[f"ci{int(confidence_level * 100)}_low_{metric_name}"] = float(low)
        summary[f"ci{int(confidence_level * 100)}_high_{metric_name}"] = float(high)

    summary["total_evaluation_examples"] = sum(
        result.evaluation.examples for result in results
    )
    return summary


def _within_subject_run_manifest(
    *,
    config: WithinSubjectConfig,
    dataset: SavedDataset,
    combination: Combination,
    subjects: list[int],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_dir.name,
        "run_directory": str(run_dir),
        "experiment": "ourexperimentversionfour",
        "diagnostic": "within_subject",
        "combination": {
            "name": combination.name,
            "node_variant": combination.node_variant,
            "edge_variant": combination.edge_variant,
            "band_name": combination.band_name,
        },
        "requested_subjects": list(subjects),
        "within_subject_config": _serializable_config(config),
        "dataset_path": str(dataset.path),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "git": _git_provenance(),
    }


def _save_within_subject_subject_outputs(
    subject_id: int,
    results: tuple[WithinSubjectFoldResult, ...],
    run_dir: Path,
) -> None:
    subject_dir = run_dir / f"subject_{subject_id:02d}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        subject_dir / "within_subject_results.json",
        {
            "subject_id": subject_id,
            "folds": [result.as_dict() for result in results],
        },
    )


def train_all_within_subject(
    config: WithinSubjectConfig,
    subject_ids: list[int] | None = None,
    *,
    dataset: SavedDataset | None = None,
    show_progress: bool = True,
) -> tuple[list[WithinSubjectFoldResult], dict[str, Any]]:
    """Run the within-subject diagnostic for every requested subject.

    ``subject_ids=None`` (the default) runs every subject in the dataset;
    passing an explicit list runs a fast few-subject pilot instead (see
    ``WITHIN_SUBJECT_PLAN.md``'s verification step: always try a small pilot
    before the full 50-subject sweep).

    Normalization is fit per fold, from that fold's training indices only
    (see :func:`train_within_subject_fold`) -- there is no dataset-wide
    normalization to fit once here, since each fold needs its own,
    differently-scoped statistics.
    """

    combination = get_combination(config.combination)
    graph_dataset = dataset if dataset is not None else combination.load_dataset()
    combination.validate_dataset(graph_dataset)

    available_subjects = tuple(
        sorted({int(sample["subject"]) for sample in graph_dataset.samples})
    )
    resolved_subject_ids = (
        list(available_subjects) if subject_ids is None else list(subject_ids)
    )
    missing = set(resolved_subject_ids) - set(available_subjects)
    if missing:
        raise ValueError(
            f"Unknown subjects {sorted(missing)}; available subjects: "
            f"{available_subjects}"
        )

    run_dir = _prepare_run_directory(config) if config.save_outputs else None
    manifest: dict[str, Any] | None = None
    if run_dir is not None:
        manifest = _within_subject_run_manifest(
            config=config,
            dataset=graph_dataset,
            combination=combination,
            subjects=resolved_subject_ids,
            run_dir=run_dir,
        )
        _write_json(run_dir / "run_manifest.json", manifest)

    _log("", enabled=show_progress)
    _log("=== Within-subject diagnostic run ===", enabled=show_progress)
    _log(f"Combination     : {combination.name}", enabled=show_progress)
    _log(f"Subjects        : {len(resolved_subject_ids)} requested", enabled=show_progress)
    _log(
        f"Folds/repeats   : folds={config.folds}, repeats={config.repeats}, "
        f"epochs={config.epochs}",
        enabled=show_progress,
    )
    _log(
        f"Optimizer       : {config.optimizer}"
        + (f" (momentum={config.momentum:g})" if config.optimizer == "sgd" else ""),
        enabled=show_progress,
    )
    _log(
        f"Outputs         : {run_dir if run_dir is not None else 'disabled'}",
        enabled=show_progress,
    )

    results: list[WithinSubjectFoldResult] = []
    subjects_progress = tqdm(
        resolved_subject_ids,
        desc="Within-subject",
        unit="subject",
        disable=not _bars_enabled(show_progress),
    )
    for subject_id in subjects_progress:
        subject_results = train_within_subject(
            subject_id,
            config,
            dataset=graph_dataset,
            combination=combination,
            show_progress=show_progress,
        )
        results.extend(subject_results)
        if config.save_outputs:
            if run_dir is None:
                raise RuntimeError("Output saving requires a prepared run directory")
            _save_within_subject_subject_outputs(subject_id, subject_results, run_dir)
        mean_balanced_accuracy = float(
            np.mean([result.evaluation.balanced_accuracy for result in subject_results])
        )
        subjects_progress.set_postfix(mean_balanced_accuracy=f"{mean_balanced_accuracy:.3f}")
        _log(
            f"Subject {subject_id:02d} complete : mean balanced accuracy="
            f"{mean_balanced_accuracy:.3f}",
            enabled=show_progress,
        )

    summary = summarize_within_subject_results(results)
    if run_dir is not None:
        _write_json(run_dir / "within_subject_summary.json", summary)
        if manifest is None:
            raise RuntimeError("A saved run requires a run manifest")
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = summary
        _write_json(run_dir / "run_manifest.json", manifest)

    _log("", enabled=show_progress)
    _log("=== Within-subject diagnostic run complete ===", enabled=show_progress)
    _log(
        f"Subjects={summary['subjects']}, "
        f"balanced_accuracy={summary['mean_balanced_accuracy']:.3f} +/- "
        f"{summary['std_balanced_accuracy']:.3f}, "
        f"accuracy={summary['mean_accuracy']:.3f} +/- "
        f"{summary['std_accuracy']:.3f}, "
        f"auc={summary['mean_auc']:.3f} +/- {summary['std_auc']:.3f}",
        enabled=show_progress,
    )
    if run_dir is not None:
        _log(f"Results saved   : {run_dir}", enabled=show_progress)
    return results, summary
