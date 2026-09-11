"""Inner-CV hyperparameter search for one LOSO outer fold's development set.

Scores candidate ``TrainingConfig`` overrides against each other using
grouped inner cross-validation (see
``src.ourexperimentversionfour.data.loso_split.create_inner_cv_folds``) over
an outer fold's non-test subjects, so the held-out test subject is never
touched by anything in this module. See ``training/loso.py``'s
``train_loso_fold_with_search`` for how the winning candidate then feeds
into an ordinary LOSO retrain + single test evaluation.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from torch import nn

from src.datautils.graphdataversionone.saved_dataset import SavedDataset
from src.ourexperimentversionfour.data import EXPECTED_NODES, EXPECTED_NODE_FEATURES
from src.ourexperimentversionfour.data.combinations import Combination
from src.ourexperimentversionfour.data.loso_split import GraphDataLoaderConfig, InnerCvFold, create_inner_cv_folds
from src.ourexperimentversionfour.model import EEGGCN1, EEGGCN1Config

from .config import TrainingConfig
from .engine import build_optimizer, evaluate, resolve_device, set_seed, train_epoch


DEFAULT_SEARCH_GRID: list[dict[str, Any]] = [
    {"learning_rate": learning_rate, "weight_decay": weight_decay}
    for learning_rate in (0.001, 0.005, 0.01, 0.02)
    for weight_decay in (1e-4, 5e-4, 1e-3)
]
"""12 candidates (4 learning rates x 3 weight decays). Easy to edit/replace;
only ``TrainingConfig`` field overrides are valid entries."""


def _score_candidate(
    combination: Combination,
    dataset: SavedDataset,
    inner_cv_folds: tuple[InnerCvFold, ...],
    overrides: dict[str, Any],
    base_config: TrainingConfig,
    *,
    search_epochs: int,
    search_patience: int | None,
    seed: int,
) -> list[float]:
    """Train one candidate on every inner fold; return its per-fold scores.

    Each inner fold's own best epoch is selected by validation **loss**
    (matching ``train_loso_fold``'s existing criterion) -- the value
    returned for that fold is the validation **balanced accuracy** recorded
    at that same loss-selected epoch, not a separately-tracked "best
    balanced accuracy epoch" (which would double-optimize on a noisier
    metric). Every inner fold uses the same base ``seed`` for model
    initialization -- the differing train/validation subject splits already
    provide the k independent estimates.
    """

    candidate_config = dataclasses.replace(
        base_config,
        **overrides,
        epochs=search_epochs,
        patience=search_patience,
        save_outputs=False,
    )
    device = resolve_device(candidate_config.device)
    loader_config = GraphDataLoaderConfig(
        batch_size=candidate_config.batch_size,
        num_workers=candidate_config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=candidate_config.num_workers > 0,
        seed=candidate_config.seed,
    )

    per_fold_balanced_accuracy: list[float] = []
    for inner_fold in inner_cv_folds:
        train_loader, validation_loader, _normalization = combination.create_group_dataloaders(
            dataset=dataset,
            train_subject_ids=inner_fold.train_subject_ids,
            validation_subject_ids=inner_fold.validation_subject_ids,
            config=loader_config,
            dataset_validated=True,
        )

        set_seed(seed)
        model = EEGGCN1(
            EEGGCN1Config(input_features=EXPECTED_NODE_FEATURES, nodes=EXPECTED_NODES)
        ).to(device)
        optimizer = build_optimizer(
            model,
            optimizer=candidate_config.optimizer,
            learning_rate=candidate_config.learning_rate,
            weight_decay=candidate_config.weight_decay,
            momentum=candidate_config.momentum,
        )
        loss_function = nn.NLLLoss()

        best_validation_loss = float("inf")
        best_validation_balanced_accuracy = 0.0
        epochs_without_improvement = 0
        for _epoch in range(1, candidate_config.epochs + 1):
            train_epoch(
                model,
                train_loader,
                loss_function,
                optimizer,
                device,
                gradient_clip_norm=candidate_config.gradient_clip_norm,
            )
            validation_metrics = evaluate(model, validation_loader, loss_function, device)
            improved = (
                validation_metrics.loss
                < best_validation_loss - candidate_config.minimum_improvement
            )
            if improved:
                best_validation_loss = validation_metrics.loss
                best_validation_balanced_accuracy = validation_metrics.balanced_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (
                candidate_config.patience is not None
                and epochs_without_improvement >= candidate_config.patience
            ):
                break

        per_fold_balanced_accuracy.append(best_validation_balanced_accuracy)

    return per_fold_balanced_accuracy


def select_hyperparameters(
    combination: Combination,
    dataset: SavedDataset,
    train_subject_ids: tuple[int, ...],
    *,
    grid: list[dict[str, Any]] | None = None,
    inner_folds: int = 10,
    search_epochs: int = 50,
    search_patience: int | None = 8,
    base_config: TrainingConfig,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every candidate in ``grid`` via inner CV; return the winner.

    ``inner_folds`` is the fold count ``k`` (not already-built folds -- see
    :func:`_score_candidate` for the per-candidate scoring, which does take
    the built ``InnerCvFold`` tuple). The same ``k`` folds, built once here,
    are reused for every candidate so comparisons vary only the
    hyperparameters, not the data split.

    Returns ``(best_overrides, all_candidate_scores)``; each entry in
    ``all_candidate_scores`` is ``{"overrides": dict, "mean_balanced_accuracy":
    float, "per_fold_balanced_accuracy": list[float]}``, kept for
    provenance/logging.
    """

    resolved_grid = grid if grid is not None else DEFAULT_SEARCH_GRID
    if not resolved_grid:
        raise ValueError("grid must contain at least one candidate")

    subject_ids = np.asarray(
        [sample["subject"] for sample in dataset.samples], dtype=np.int64
    )
    inner_cv_folds = create_inner_cv_folds(
        subject_ids, train_subject_ids, k=inner_folds, seed=seed
    )

    all_candidate_scores: list[dict[str, Any]] = []
    best_overrides: dict[str, Any] | None = None
    best_mean_score = float("-inf")
    for overrides in resolved_grid:
        per_fold_scores = _score_candidate(
            combination,
            dataset,
            inner_cv_folds,
            overrides,
            base_config,
            search_epochs=search_epochs,
            search_patience=search_patience,
            seed=seed,
        )
        mean_score = float(np.mean(per_fold_scores))
        all_candidate_scores.append(
            {
                "overrides": overrides,
                "mean_balanced_accuracy": mean_score,
                "per_fold_balanced_accuracy": per_fold_scores,
            }
        )
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_overrides = overrides

    assert best_overrides is not None
    return best_overrides, all_candidate_scores
