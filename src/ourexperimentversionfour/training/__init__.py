"""Training utilities for the without-CSD alpha-band wPLI experiment."""

from .config import DEFAULT_OUTPUT_DIR, SeedStrategy, TrainingConfig
from .engine import evaluate, resolve_device, set_seed, train_epoch
from .hyperparameter_search import DEFAULT_SEARCH_GRID, select_hyperparameters
from .loso import (
    EpochRecord,
    FoldResult,
    summarize_results,
    train_all_loso_folds,
    train_all_loso_folds_with_search,
    train_loso_fold,
    train_loso_fold_with_search,
)
from .metrics import ClassificationMetrics

__all__ = [
    "ClassificationMetrics",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SEARCH_GRID",
    "EpochRecord",
    "FoldResult",
    "SeedStrategy",
    "TrainingConfig",
    "evaluate",
    "resolve_device",
    "select_hyperparameters",
    "set_seed",
    "summarize_results",
    "train_all_loso_folds",
    "train_all_loso_folds_with_search",
    "train_epoch",
    "train_loso_fold",
    "train_loso_fold_with_search",
]
