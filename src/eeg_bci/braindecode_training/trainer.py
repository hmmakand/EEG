from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

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
    latest_history_value,
    score_classifier,
    score_classifier_by_description,
)
from eeg_bci.data.splitting import LEAVE_ONE_SUBJECT_OUT
from eeg_bci.data.splitting import SESSION_CROSS_VALIDATION_TEST
from eeg_bci.data.splitting import SESSION_GRID_SEARCH_TEST
from eeg_bci.data.splitting import SESSION_TRAIN_TEST
from eeg_bci.data.splitting import SESSION_TRAIN_VALID_TEST
from eeg_bci.data.splitting import SplitPlan, split_train_valid
from eeg_bci.tracking.artifacts import export_history
from eeg_bci.tracking.tensorboard import write_final_scalars, write_history_scalars

MetricValue = float | int | str
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
) -> dict[str, MetricValue]:
    if split_plan.split_strategy in {
        SESSION_TRAIN_TEST,
        SESSION_TRAIN_VALID_TEST,
        LEAVE_ONE_SUBJECT_OUT,
        "random",
    }:
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
        )

    if split_plan.split_strategy == SESSION_CROSS_VALIDATION_TEST:
        return _train_with_cross_validation(
            model,
            split_plan,
            n_outputs=n_outputs,
            device=device,
            training_cfg=training_cfg,
            output_dir=output_dir,
            tensorboard_dir=tensorboard_dir,
        )

    if split_plan.split_strategy == SESSION_GRID_SEARCH_TEST:
        return _train_with_grid_search(
            model,
            split_plan,
            n_outputs=n_outputs,
            device=device,
            training_cfg=training_cfg,
            output_dir=output_dir,
            tensorboard_dir=tensorboard_dir,
        )

    raise ValueError(f"Unsupported split strategy {split_plan.split_strategy}.")


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
) -> dict[str, MetricValue]:
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
    )


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
    test_acc = score_classifier(classifier, test_set)
    train_loss = latest_history_value(classifier, "train_loss")
    train_acc = latest_history_value(classifier, "train_accuracy")
    save_classifier_module(classifier, output_dir, str(training_cfg.checkpoint_name))
    history_path = export_history(classifier, output_dir)
    if tensorboard_dir is not None:
        write_history_scalars(tensorboard_dir, classifier)
        write_final_scalars(tensorboard_dir, {"test_acc": test_acc})
    subject_results_path = _write_subject_test_results(classifier, test_set, output_dir)

    metrics: dict[str, MetricValue] = {
        "split_strategy": split_strategy,
        "n_train_windows": len(train_set),
        "n_valid_windows": len(valid_set) if valid_set is not None else 0,
        "n_test_windows": len(test_set),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "checkpoint_path": str(output_dir / "checkpoints" / str(training_cfg.checkpoint_name)),
        "history_path": str(history_path),
    }
    if valid_set is not None:
        metrics["valid_loss"] = latest_history_value(classifier, "valid_loss")
        metrics["valid_acc"] = latest_history_value(classifier, "valid_accuracy")
    if subject_results_path is not None:
        metrics["subject_test_results"] = str(subject_results_path)

    logger.info("test_acc=%.4f", test_acc)
    if subject_results_path is not None:
        logger.info("subject_test_results=%s", subject_results_path)
    return metrics


def _train_with_cross_validation(
    model: nn.Module,
    split_plan: SplitPlan,
    *,
    n_outputs: int,
    device: torch.device,
    training_cfg: DictConfig,
    output_dir: Path,
    tensorboard_dir: Path | None,
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
    test_acc = score_classifier(final_classifier, split_plan.test_set)
    train_loss = latest_history_value(final_classifier, "train_loss")
    train_acc = latest_history_value(final_classifier, "train_accuracy")
    save_classifier_module(final_classifier, output_dir, str(training_cfg.checkpoint_name))
    history_path = export_history(final_classifier, output_dir)
    subject_results_path = _write_subject_test_results(
        final_classifier,
        split_plan.test_set,
        output_dir,
    )

    metrics: dict[str, MetricValue] = {
        "split_strategy": split_plan.split_strategy,
        "n_train_windows": len(split_plan.train_pool),
        "n_valid_windows": 0,
        "n_test_windows": len(split_plan.test_set),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "cv_acc_mean": float(np.mean(cv_scores)),
        "cv_acc_std": float(np.std(cv_scores)),
        "test_acc": test_acc,
        "checkpoint_path": str(output_dir / "checkpoints" / str(training_cfg.checkpoint_name)),
        "history_path": str(history_path),
    }
    metrics.update(_fold_score_metrics("cv_acc", cv_scores))
    if subject_results_path is not None:
        metrics["subject_test_results"] = str(subject_results_path)
    if tensorboard_dir is not None:
        write_history_scalars(tensorboard_dir, final_classifier)
        write_final_scalars(tensorboard_dir, metrics)

    logger.info("cv_acc_mean=%.4f", float(metrics["cv_acc_mean"]))
    logger.info("test_acc=%.4f", test_acc)
    if subject_results_path is not None:
        logger.info("subject_test_results=%s", subject_results_path)
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
    test_acc = score_classifier(best_classifier, split_plan.test_set)
    train_loss = latest_history_value(best_classifier, "train_loss")
    train_acc = latest_history_value(best_classifier, "train_accuracy")
    save_classifier_module(best_classifier, output_dir, str(training_cfg.checkpoint_name))
    history_path = export_history(best_classifier, output_dir)
    subject_results_path = _write_subject_test_results(
        best_classifier,
        split_plan.test_set,
        output_dir,
    )

    metrics: dict[str, MetricValue] = {
        "split_strategy": split_plan.split_strategy,
        "n_train_windows": len(split_plan.train_pool),
        "n_valid_windows": 0,
        "n_test_windows": len(split_plan.test_set),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "best_score": float(search.best_score_),
        "best_params": str(search.best_params_),
        "test_acc": test_acc,
        "checkpoint_path": str(output_dir / "checkpoints" / str(training_cfg.checkpoint_name)),
        "history_path": str(history_path),
    }
    metrics.update(_grid_search_summary_metrics(search.cv_results_, int(search.best_index_)))
    if subject_results_path is not None:
        metrics["subject_test_results"] = str(subject_results_path)
    if tensorboard_dir is not None:
        write_history_scalars(tensorboard_dir, best_classifier)
        write_final_scalars(tensorboard_dir, metrics)

    logger.info("best_score=%.4f", float(metrics["best_score"]))
    logger.info("test_acc=%.4f", test_acc)
    if subject_results_path is not None:
        logger.info("subject_test_results=%s", subject_results_path)
    return metrics


def _write_subject_test_results(
    classifier: Any,
    test_set: Dataset,
    output_dir: Path,
) -> Path | None:
    rows = score_classifier_by_description(
        classifier,
        test_set,
        description_key="subject",
    )
    if len(rows) < 2:
        return None

    path = output_dir / "results" / "subject_pooled_test_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["subject", "n_windows", "test_acc"])
        writer.writeheader()
        writer.writerows(rows)
    return path


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
