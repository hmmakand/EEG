"""Within-subject classification: a diagnostic sanity check alongside LOSO.

Trains and evaluates on one subject's own trials only, with no cross-subject
generalization involved -- see ``WITHIN_SUBJECT_PLAN.md`` for the full
rationale (isolating "can the model fit this data at all" and "is there
decodable signal at all" from the harder cross-subject LOSO problem).

Mirrors ``hyperparameter_search.py``'s reuse of ``engine.py``'s low-level
training/evaluation building blocks and ``EEGGCN1`` exactly as
``loso.py`` does; only fold construction, config, and orchestration differ
here, since the two evaluation protocols do not share a shape (one
subject's trials per fold rather than a cross-subject group). Generic
(non-LOSO-specific) provenance/logging/run-directory helpers are imported
from ``.loso`` rather than duplicated.

Each fold trains twice (see ``train_within_subject_fold``): a selection pass
that early-stops on an inner validation split carved out of the fold's own
training trials (never its evaluation trials) to pick an epoch count, then a
final pass that fits all of the fold's training trials for exactly that many
epochs before the one, real evaluation. This replaced an earlier design that
checkpointed on training loss alone, which -- because training loss
decreases near-monotonically -- always selected the most-overfit, latest
epoch and produced chance-level held-out accuracy despite >90% training
accuracy.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm, trange

from src.datautils.graphdataversiontwo.saved_dataset import SavedDataset
from src.ourexperimentversionfive.data import (
    EXPECTED_NODES,
    EXPECTED_NODE_FEATURES,
    GraphDataLoaderConfig,
)
from src.ourexperimentversionfive.data.combinations import Combination, get_combination
from src.ourexperimentversionfive.data.within_subject_split import (
    WithinSubjectFold,
    create_inner_validation_split,
    create_within_subject_folds,
)
from src.ourexperimentversionfive.data.validation import feature_provenance
from src.ourexperimentversionfive.model import EEGGCN1, EEGGCN1Config

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
    """Final-pass evaluation result and history for one within-subject fold.

    ``best_epoch`` is the epoch count selected by the inner-validation
    selection pass (see ``train_within_subject_fold``) -- the final pass
    always trains for exactly this many epochs, so ``epochs_ran`` equals it.
    ``inner_selection`` records the selection pass's own diagnostics
    (never used to compute ``evaluation``, which only ever sees the final
    pass's model).
    """

    subject_id: int
    fold: int
    repeat: int
    best_epoch: int
    epochs_ran: int
    evaluation: ClassificationMetrics
    history: tuple[WithinSubjectEpochRecord, ...]
    preprocessing: dict[str, Any] | None = None
    inner_selection: dict[str, Any] | None = None
    final_training: ClassificationMetrics | None = None

    def generalization_gap(self) -> dict[str, float] | None:
        """Positive values mean worse held-out performance (loss has reversed sign)."""
        if self.final_training is None:
            return None
        return {
            "accuracy": self.final_training.accuracy - self.evaluation.accuracy,
            "balanced_accuracy": self.final_training.balanced_accuracy - self.evaluation.balanced_accuracy,
            "loss": self.evaluation.loss - self.final_training.loss,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "fold": self.fold,
            "repeat": self.repeat,
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "evaluation": self.evaluation.as_dict(),
            "history": [record.as_dict() for record in self.history],
            "preprocessing": self.preprocessing,
            "inner_selection": self.inner_selection,
            "final_training": self.final_training.as_dict() if self.final_training else None,
            "generalization_gap": self.generalization_gap(),
        }


def _dataset_subject_ids(dataset: SavedDataset) -> np.ndarray:
    return np.asarray([sample["subject"] for sample in dataset.samples], dtype=np.int64)


def _run_selection_pass(
    subject_id: int,
    fold: WithinSubjectFold,
    *,
    combination: Combination,
    dataset: SavedDataset,
    config: WithinSubjectConfig,
    loader_config: GraphDataLoaderConfig,
    device: torch.device,
    loss_function: nn.Module,
    repeat: int,
    show_progress: bool,
) -> dict[str, Any]:
    """Pick an epoch count by early-stopping on an inner validation split.

    The inner split is drawn only from ``fold.train_graph_indices`` (never
    ``fold.evaluation_graph_indices``), so nothing here can leak into the
    reported evaluation result. Mirrors ``train_loso_fold``'s
    validation-loss/patience criterion; the only difference is *what* gets
    held out (a slice of one subject's training trials, since there are no
    spare subjects to hold out here).
    """

    labels = np.asarray(dataset.labels)
    inner_train_graph_indices, inner_validation_graph_indices = (
        create_inner_validation_split(
            fold.train_graph_indices,
            labels,
            validation_fraction=config.inner_validation_fraction,
            seed=config.seed + 1000 * repeat + fold.fold,
        )
    )
    inner_train_loader, inner_validation_loader, _inner_normalization = (
        combination.create_within_subject_dataloaders(
            dataset=dataset,
            train_graph_indices=inner_train_graph_indices,
            evaluation_graph_indices=inner_validation_graph_indices,
            config=loader_config,
            dataset_validated=True,
        )
    )

    set_seed(config.seed)
    model = EEGGCN1(
        EEGGCN1Config(input_features=EXPECTED_NODE_FEATURES, nodes=EXPECTED_NODES,
                      edge_mode=config.edge_mode, classifier=config.classifier)
    ).to(device)
    optimizer = build_optimizer(
        model,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )

    best_validation_loss = float("inf")
    selected_epochs = 0
    epochs_without_improvement = 0
    epochs_ran = 0
    selection_history: list[dict[str, Any]] = []

    progress = trange(
        1,
        config.epochs + 1,
        desc=f"Subject {subject_id:02d} fold {fold.fold} repeat {repeat} (select)",
        unit="epoch",
        disable=not _bars_enabled(show_progress),
        leave=False,
    )
    for epoch in progress:
        training_metrics = train_epoch(
            model,
            inner_train_loader,
            loss_function,
            optimizer,
            device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_metrics = evaluate(model, inner_validation_loader, loss_function, device)
        selection_history.append({
            "epoch": epoch,
            "training_online": training_metrics.as_dict(),
            "validation": validation_metrics.as_dict(),
        })
        epochs_ran = epoch
        improved = (
            validation_metrics.loss < best_validation_loss - config.minimum_improvement
        )
        if improved:
            best_validation_loss = validation_metrics.loss
            selected_epochs = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        progress.set_postfix(
            inner_val_loss=f"{validation_metrics.loss:.4f}",
            best=f"{best_validation_loss:.4f}@{selected_epochs}",
        )
        _log(
            f"Subject {subject_id:02d} fold {fold.fold} repeat {repeat} | "
            f"select epoch {epoch:03d}/{config.epochs:03d} | "
            f"inner val loss={validation_metrics.loss:.4f} "
            f"acc={validation_metrics.accuracy:.3f} | "
            f"best={best_validation_loss:.4f}@{selected_epochs}",
            enabled=show_progress,
        )
        if (
            config.patience is not None
            and epochs_without_improvement >= config.patience
        ):
            _log(
                f"Subject {subject_id:02d} fold {fold.fold} repeat {repeat} | "
                f"selection early stopping: no inner-validation-loss improvement "
                f"for {config.patience} epochs",
                enabled=show_progress,
            )
            break

    if selected_epochs == 0:
        raise RuntimeError("Inner validation selection did not produce a best epoch")

    return {
        "epochs_ran": epochs_ran,
        "selected_epoch": selected_epochs,
        "best_validation_loss": best_validation_loss,
        "history": selection_history,
        "inner_train_graph_indices": inner_train_graph_indices.tolist(),
        "inner_validation_graph_indices": inner_validation_graph_indices.tolist(),
        "inner_train_examples": int(len(inner_train_graph_indices)),
        "inner_validation_examples": int(len(inner_validation_graph_indices)),
    }


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
    """Select an epoch budget on an inner split, then fit and evaluate once.

    Two training passes, neither of which lets ``fold.evaluation_graph_indices``
    influence anything but the final, reported metrics:

    1. **Selection pass** (:func:`_run_selection_pass`) -- train on an inner
       slice of this fold's training trials, early-stop on the remaining
       inner-validation slice's loss, and record the epoch that produced the
       best inner-validation loss.
    2. **Final pass** -- re-initialize the model and train on *all* of this
       fold's training trials for exactly that many epochs, no
       checkpointing needed since the epoch count was already chosen by a
       genuine held-out signal. That final state is evaluated once on
       ``fold.evaluation_graph_indices``.

    This replaced checkpointing on **training** loss (zero-leakage, but
    training loss decreases near-monotonically, so it always picked the
    most-overfit, latest epoch and produced chance-level held-out accuracy
    despite >90% training accuracy).

    Optional z-scoring for the final pass is fitted only on this fold's
    training trials, as in LOSO and inner search; the selection pass fits
    its own normalization from its inner-train trials alone, since it never
    touches ``fold.evaluation_graph_indices``.
    """

    device = resolve_device(config.device)
    loader_config = GraphDataLoaderConfig(
        node_normalization=config.node_normalization,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        seed=config.seed,
    )
    loss_function = nn.NLLLoss()

    inner_selection = _run_selection_pass(
        subject_id,
        fold,
        combination=combination,
        dataset=dataset,
        config=config,
        loader_config=loader_config,
        device=device,
        loss_function=loss_function,
        repeat=repeat,
        show_progress=show_progress,
    )
    selected_epochs = inner_selection["selected_epoch"]

    fold_normalization = combination.fit_feature_normalization(
        dataset,
        epsilon=loader_config.normalization_epsilon,
        mode=config.node_normalization,
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
        EEGGCN1Config(input_features=EXPECTED_NODE_FEATURES, nodes=EXPECTED_NODES,
                      edge_mode=config.edge_mode, classifier=config.classifier)
    ).to(device)
    optimizer = build_optimizer(
        model,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )

    history: list[WithinSubjectEpochRecord] = []
    epochs = trange(
        1,
        selected_epochs + 1,
        desc=f"Subject {subject_id:02d} fold {fold.fold} repeat {repeat} (fit)",
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
        epochs.set_postfix(tr_loss=f"{training_metrics.loss:.4f}")
        _log(
            f"Subject {subject_id:02d} fold {fold.fold} repeat {repeat} | "
            f"fit epoch {epoch:03d}/{selected_epochs:03d} | "
            f"train loss={training_metrics.loss:.4f} "
            f"acc={training_metrics.accuracy:.3f}",
            enabled=show_progress,
        )

    evaluation_metrics = evaluate(model, evaluation_loader, loss_function, device)
    # Fixed final weights, dropout disabled, frozen BatchNorm statistics.
    # Run after held-out evaluation; this diagnostic cannot affect selection.
    final_training = evaluate(model, train_loader, loss_function, device)
    _log(
        f"Subject {subject_id:02d} fold {fold.fold} repeat {repeat} | "
        f"final train/evaluation balanced accuracy="
        f"{final_training.balanced_accuracy:.3f}/{evaluation_metrics.balanced_accuracy:.3f} | "
        f"gap={final_training.balanced_accuracy - evaluation_metrics.balanced_accuracy:+.3f} | "
        f"final train/evaluation loss={final_training.loss:.4f}/{evaluation_metrics.loss:.4f}",
        enabled=show_progress,
    )
    return WithinSubjectFoldResult(
        subject_id=subject_id,
        fold=fold.fold,
        repeat=repeat,
        best_epoch=selected_epochs,
        epochs_ran=len(history),
        evaluation=evaluation_metrics,
        final_training=final_training,
        history=tuple(history),
        preprocessing={
            "mode": fold_normalization.mode,
            "epsilon": fold_normalization.epsilon,
            "fit_graph_indices": list(fold_normalization.fit_graph_indices),
            "train_graph_indices": fold.train_graph_indices.tolist(),
            "evaluation_graph_indices": fold.evaluation_graph_indices.tolist(),
            "mean": fold_normalization.mean_by_subject[subject_id].tolist() if fold_normalization.mode == "zscore" else None,
            "standard_deviation": fold_normalization.standard_deviation_by_subject[subject_id].tolist() if fold_normalization.mode == "zscore" else None,
        },
        inner_selection=inner_selection,
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

    # Match the existing equal-subject weighting; old manually constructed
    # results without diagnostics remain supported.
    diagnostic_results = [r for r in results if r.final_training is not None]
    if diagnostic_results:
        diagnostic_subjects = sorted({r.subject_id for r in diagnostic_results})
        diagnostics: dict[str, Any] = {"subjects": len(diagnostic_subjects),
                                       "folds": len(diagnostic_results)}
        for metric in ("accuracy", "balanced_accuracy", "loss"):
            diagnostics[f"mean_final_training_{metric}"] = float(np.mean([
                np.mean([getattr(r.final_training, metric) for r in diagnostic_results
                         if r.subject_id == subject]) for subject in diagnostic_subjects
            ]))
            diagnostics[f"mean_gap_{metric}"] = float(np.mean([
                np.mean([r.generalization_gap()[metric] for r in diagnostic_results
                         if r.subject_id == subject]) for subject in diagnostic_subjects
            ]))
        summary["diagnostics"] = diagnostics

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
        "experiment": "ourexperimentversionfive",
        "feature_selection": feature_provenance(dataset),
        "classifier": config.classifier,
        "edge_mode": config.edge_mode,
        "node_normalization": config.node_normalization,
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
