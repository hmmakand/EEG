from __future__ import annotations

import csv
import logging
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GridSearchCV, cross_val_score
from skorch.helper import SliceDataset
from torch import nn
from torch.utils.data import Dataset

from eeg_bci.braindecode_training.checkpointing import save_classifier_module
from eeg_bci.braindecode_training.classifier import build_eeg_classifier
from eeg_bci.braindecode_training.evaluation import (
    evaluate_classifier,
    latest_history_value,
    score_classifier_by_description,
)
from eeg_bci.data.splitting import (
    CROSS_VALIDATION_TEST,
    GRID_SEARCH_TEST,
    LOSO,
    TRAIN_TEST,
    TRAIN_VALID_TEST,
    SplitPlan,
    split_train_valid,
)
from eeg_bci.tracking.artifacts import export_history
from eeg_bci.tracking.metrics import DEFAULT_METRICS, normalize_eval_metrics
from eeg_bci.tracking.results import order_fieldnames
from eeg_bci.tracking.tensorboard import (
    write_confusion_matrix,
    write_final_scalars,
    write_history_scalars,
)

MetricValue = Any
logger = logging.getLogger(__name__)


def train_from_split_plan(
    model: nn.Module,
    split_plan: SplitPlan,
    *,
    n_outputs: int,
    device: torch.device,
    training_cfg: DictConfig,
    output_dir: Path,
    tensorboard_dir: Path | None = None,
    class_names: list[str] | None = None,
) -> dict[str, MetricValue]:
    eval_metrics_cfg = _eval_metrics_cfg(training_cfg)
    labels = _evaluation_labels(n_outputs)
    _validate_class_names(class_names, labels)

    if split_plan.method in {TRAIN_TEST, TRAIN_VALID_TEST, LOSO}:
        return _train_once(
            model,
            split_plan.train_set,
            split_plan.test_set,
            valid_set=split_plan.valid_set,
            n_outputs=n_outputs,
            device=device,
            training_cfg=training_cfg,
            output_dir=output_dir,
            tensorboard_dir=tensorboard_dir,
            split_strategy=split_plan.split_strategy,
            eval_metrics=eval_metrics_cfg,
            labels=labels,
            class_names=class_names,
        )

    if split_plan.method == CROSS_VALIDATION_TEST:
        return _train_with_cross_validation(
            model,
            split_plan,
            n_outputs=n_outputs,
            device=device,
            training_cfg=training_cfg,
            output_dir=output_dir,
            tensorboard_dir=tensorboard_dir,
            eval_metrics=eval_metrics_cfg,
            labels=labels,
            class_names=class_names,
        )

    if split_plan.method == GRID_SEARCH_TEST:
        return _train_with_grid_search(
            model,
            split_plan,
            n_outputs=n_outputs,
            device=device,
            training_cfg=training_cfg,
            output_dir=output_dir,
            tensorboard_dir=tensorboard_dir,
            eval_metrics=eval_metrics_cfg,
            labels=labels,
            class_names=class_names,
        )

    raise ValueError(f"Unsupported split method {split_plan.method}.")


def train_model(
    model: nn.Module,
    train_set: Dataset,
    test_set: Dataset,
    *,
    n_outputs: int,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    output_dir: Path,
    checkpoint_name: str,
    validation_enabled: bool,
    validation_size: float,
    validation_shuffle: bool,
    seed: int,
    tensorboard_dir: Path | None = None,
    class_names: list[str] | None = None,
) -> dict[str, MetricValue]:
    labels = _evaluation_labels(n_outputs)
    _validate_class_names(class_names, labels)

    valid_set: Dataset | None = None
    if validation_enabled:
        train_set, valid_set = split_train_valid(
            train_set,
            valid_size=validation_size,
            shuffle=validation_shuffle,
            seed=seed,
        )

    training_cfg = OmegaConf.create(
        {
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_workers": num_workers,
            "checkpoint_name": checkpoint_name,
        }
    )
    return _train_once(
        model,
        train_set,
        test_set,
        valid_set=valid_set,
        n_outputs=n_outputs,
        device=device,
        training_cfg=training_cfg,
        output_dir=output_dir,
        tensorboard_dir=tensorboard_dir,
        split_strategy="single_fit",
        eval_metrics=_eval_metrics_cfg(training_cfg),
        labels=labels,
        class_names=class_names,
    )


def _persist_and_collect(
    classifier: Any,
    test_set: Dataset,
    output_dir: Path,
    tensorboard_dir: Path | None,
    *,
    split_strategy: str,
    n_train_windows: int,
    n_valid_windows: int,
    n_test_windows: int,
    checkpoint_name: str,
    eval_metrics: list[str],
    labels: list[int],
    class_names: list[str] | None,
    extra: dict[str, MetricValue] | None = None,
) -> dict[str, MetricValue]:
    """Evaluate, checkpoint, persist artifacts, and assemble the metrics dict.

    This is the shared tail of every training path (single fit, cross-validation,
    grid search). ``extra`` carries path-specific metrics (``valid_*``, ``cv_*``,
    ``best_*``).
    """

    test_metrics = evaluate_classifier(
        classifier,
        test_set,
        metrics=eval_metrics,
        class_names=class_names,
        labels=labels,
    )
    test_accuracy = float(test_metrics["accuracy"])

    save_classifier_module(classifier, output_dir, checkpoint_name)
    history_path = export_history(classifier, output_dir)
    subject_results_path = _write_subject_test_results(
        classifier,
        test_set,
        output_dir,
        metrics=eval_metrics,
        labels=labels,
        class_names=class_names,
    )

    metrics: dict[str, MetricValue] = {
        "split_strategy": split_strategy,
        "n_train_windows": n_train_windows,
        "n_valid_windows": n_valid_windows,
        "n_test_windows": n_test_windows,
        "train_loss": latest_history_value(classifier, "train_loss"),
        "train_accuracy": latest_history_value(classifier, "train_accuracy"),
        **_prefix_test_metrics(test_metrics),
        "checkpoint_path": str(output_dir / "checkpoints" / checkpoint_name),
        "history_path": str(history_path),
    }
    if extra:
        metrics.update(extra)
    if subject_results_path is not None:
        metrics["subject_test_results"] = str(subject_results_path)

    if tensorboard_dir is not None:
        write_history_scalars(tensorboard_dir, classifier)
        write_final_scalars(tensorboard_dir, metrics)
        _write_confusion_matrix_if_present(tensorboard_dir, test_metrics, class_names=class_names)

    logger.info("test_accuracy=%.4f", test_accuracy)
    if subject_results_path is not None:
        logger.info("subject_test_results=%s", subject_results_path)
    return metrics


def _train_once(
    model: nn.Module,
    train_set: Dataset,
    test_set: Dataset,
    *,
    valid_set: Dataset | None,
    n_outputs: int,
    device: torch.device,
    training_cfg: DictConfig,
    output_dir: Path,
    tensorboard_dir: Path | None,
    split_strategy: str,
    eval_metrics: list[str],
    labels: list[int],
    class_names: list[str] | None = None,
) -> dict[str, MetricValue]:
    classifier = _build_classifier(
        model,
        n_outputs,
        device,
        training_cfg,
        valid_set=valid_set,
        tensorboard_dir=tensorboard_dir,
    )
    classifier.fit(train_set, y=None)

    extra: dict[str, MetricValue] = {}
    if valid_set is not None:
        extra["valid_loss"] = latest_history_value(classifier, "valid_loss")
        extra["valid_accuracy"] = latest_history_value(classifier, "valid_accuracy")

    return _persist_and_collect(
        classifier,
        test_set,
        output_dir,
        tensorboard_dir,
        split_strategy=split_strategy,
        n_train_windows=_dataset_size(train_set),
        n_valid_windows=_dataset_size(valid_set) if valid_set is not None else 0,
        n_test_windows=_dataset_size(test_set),
        checkpoint_name=str(training_cfg.checkpoint_name),
        eval_metrics=eval_metrics,
        labels=labels,
        class_names=class_names,
        extra=extra,
    )


def _train_with_cross_validation(
    model: nn.Module,
    split_plan: SplitPlan,
    *,
    n_outputs: int,
    device: torch.device,
    training_cfg: DictConfig,
    output_dir: Path,
    tensorboard_dir: Path | None,
    eval_metrics: list[str],
    labels: list[int],
    class_names: list[str] | None = None,
) -> dict[str, MetricValue]:
    if split_plan.resampler is None:
        raise ValueError("Cross-validation requires a resampler.")

    classifier = _build_classifier(
        model,
        n_outputs,
        device,
        training_cfg,
        valid_set=None,
        tensorboard_dir=None,
    )
    X_train, y_train = _slice_xy(split_plan.train_pool)
    cv_scores = cross_val_score(
        classifier,
        X_train,
        y_train,
        scoring=str(training_cfg.cross_validation.scoring),
        cv=split_plan.resampler,
        n_jobs=int(training_cfg.cross_validation.n_jobs),
    )

    final_classifier = _build_classifier(
        model,
        n_outputs,
        device,
        training_cfg,
        valid_set=None,
        tensorboard_dir=tensorboard_dir,
    )
    final_classifier.fit(split_plan.train_pool, y=None)

    extra: dict[str, MetricValue] = {
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        **_fold_score_metrics("cv_accuracy", cv_scores),
    }
    metrics = _persist_and_collect(
        final_classifier,
        split_plan.test_set,
        output_dir,
        tensorboard_dir,
        split_strategy=split_plan.split_strategy,
        n_train_windows=_dataset_size(split_plan.train_pool),
        n_valid_windows=0,
        n_test_windows=_dataset_size(split_plan.test_set),
        checkpoint_name=str(training_cfg.checkpoint_name),
        eval_metrics=eval_metrics,
        labels=labels,
        class_names=class_names,
        extra=extra,
    )
    logger.info("cv_accuracy_mean=%.4f", float(metrics["cv_accuracy_mean"]))
    return metrics


def _train_with_grid_search(
    model: nn.Module,
    split_plan: SplitPlan,
    *,
    n_outputs: int,
    device: torch.device,
    training_cfg: DictConfig,
    output_dir: Path,
    tensorboard_dir: Path | None,
    eval_metrics: list[str],
    labels: list[int],
    class_names: list[str] | None = None,
) -> dict[str, MetricValue]:
    if split_plan.resampler is None:
        raise ValueError("Grid search requires a resampler.")

    classifier = _build_classifier(
        model,
        n_outputs,
        device,
        training_cfg,
        valid_set=None,
        tensorboard_dir=None,
    )
    X_train, y_train = _slice_xy(split_plan.train_pool)
    search = GridSearchCV(
        estimator=classifier,
        param_grid=_param_grid(training_cfg),
        cv=split_plan.resampler,
        return_train_score=True,
        scoring=str(training_cfg.grid_search.scoring),
        refit=bool(training_cfg.grid_search.refit),
        verbose=int(training_cfg.grid_search.verbose),
        error_score=str(training_cfg.grid_search.error_score),
        n_jobs=int(training_cfg.grid_search.n_jobs),
    )
    search.fit(X_train, y_train)

    best_classifier = search.best_estimator_
    extra: dict[str, MetricValue] = {
        "best_score": float(search.best_score_),
        "best_params": str(search.best_params_),
        **_grid_search_summary_metrics(search.cv_results_, int(search.best_index_)),
    }
    metrics = _persist_and_collect(
        best_classifier,
        split_plan.test_set,
        output_dir,
        tensorboard_dir,
        split_strategy=split_plan.split_strategy,
        n_train_windows=_dataset_size(split_plan.train_pool),
        n_valid_windows=0,
        n_test_windows=_dataset_size(split_plan.test_set),
        checkpoint_name=str(training_cfg.checkpoint_name),
        eval_metrics=eval_metrics,
        labels=labels,
        class_names=class_names,
        extra=extra,
    )
    logger.info("best_score=%.4f", float(metrics["best_score"]))
    return metrics


def _dataset_size(dataset: Dataset) -> int:
    return len(cast(Sized, dataset))


def _eval_metrics_cfg(training_cfg: DictConfig) -> list[str]:
    """Read and validate the list of evaluation metrics from the training config."""
    raw_metrics = training_cfg.get("eval_metrics", DEFAULT_METRICS)
    metrics = OmegaConf.to_container(raw_metrics, resolve=True)
    return normalize_eval_metrics(metrics)


def _evaluation_labels(n_outputs: int) -> list[int]:
    if n_outputs < 1:
        raise ValueError("n_outputs must be at least 1.")
    return list(range(n_outputs))


def _validate_class_names(class_names: list[str] | None, labels: list[int]) -> None:
    if class_names is not None and len(class_names) != len(labels):
        raise ValueError(
            f"class_names length ({len(class_names)}) must match n_outputs ({len(labels)})."
        )


def _prefix_test_metrics(test_metrics: dict[str, Any]) -> dict[str, Any]:
    """Prefix test-metric keys for persisted artifacts."""
    return {
        key if key.startswith("test_") else f"test_{key}": value
        for key, value in test_metrics.items()
    }


def _write_confusion_matrix_if_present(
    tensorboard_dir: Path,
    test_metrics: dict[str, Any],
    *,
    class_names: list[str] | None,
) -> None:
    if "confusion_matrix" in test_metrics:
        write_confusion_matrix(
            tensorboard_dir,
            test_metrics["confusion_matrix"],
            class_names=class_names,
        )


def _write_subject_test_results(
    classifier: Any,
    test_set: Dataset,
    output_dir: Path,
    metrics: list[str] | None = None,
    labels: list[int] | None = None,
    class_names: list[str] | None = None,
) -> Path | None:
    rows = score_classifier_by_description(
        classifier,
        test_set,
        description_key="subject",
        metrics=metrics,
        class_names=class_names,
        labels=labels,
    )
    if len(rows) < 2:
        return None

    rows = [_prefix_group_metrics(row, group_keys={"subject", "n_windows"}) for row in rows]
    path = output_dir / "results" / "subject_pooled_test_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _dynamic_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _prefix_group_metrics(row: dict[str, Any], *, group_keys: set[str]) -> dict[str, Any]:
    prefixed: dict[str, Any] = {}
    for key, value in row.items():
        if key in group_keys:
            prefixed[key] = value
        else:
            prefixed.update(_prefix_test_metrics({key: value}))
    return prefixed


def _dynamic_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return order_fieldnames(rows, leading=["subject", "n_windows"])


def _build_classifier(
    model: nn.Module,
    n_outputs: int,
    device: torch.device,
    training_cfg: DictConfig,
    *,
    valid_set: Dataset | None,
    tensorboard_dir: Path | None,
):
    return build_eeg_classifier(
        model,
        n_outputs=n_outputs,
        device=device,
        max_epochs=int(training_cfg.max_epochs),
        batch_size=int(training_cfg.batch_size),
        learning_rate=float(training_cfg.learning_rate),
        weight_decay=float(training_cfg.weight_decay),
        num_workers=int(training_cfg.num_workers),
        valid_set=valid_set,
        tensorboard_dir=tensorboard_dir,
    )


def _slice_xy(dataset: Dataset) -> tuple[Any, np.ndarray]:
    X: Any = SliceDataset(dataset, idx=0)
    y = np.asarray([target for target in SliceDataset(dataset, idx=1)])
    return X, y


def _param_grid(training_cfg: DictConfig) -> dict[str, Any]:
    raw_param_grid = OmegaConf.to_container(training_cfg.grid_search.param_grid, resolve=True)
    if not isinstance(raw_param_grid, dict):
        raise TypeError("training.grid_search.param_grid must be a mapping.")

    param_grid: dict[str, Any] = {}
    for key, value in raw_param_grid.items():
        if not isinstance(key, str):
            raise TypeError("training.grid_search.param_grid keys must be strings.")
        param_grid[key] = value
    return param_grid


def _fold_score_metrics(prefix: str, scores: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_fold_{index}": float(score) for index, score in enumerate(scores, 1)}


def _grid_search_summary_metrics(
    cv_results: dict[str, Any],
    best_index: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for source_key, metric_key in {
        "std_test_score": "best_score_std",
        "mean_train_score": "best_train_score",
        "std_train_score": "best_train_score_std",
    }.items():
        if source_key in cv_results:
            metrics[metric_key] = float(cv_results[source_key][best_index])
    return metrics
