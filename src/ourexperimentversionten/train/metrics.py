"""Classification metrics and summaries of selected fold results."""
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def summarize_fold_scores(fold_scores: Sequence[float]) -> dict[str, float]:
    if len(fold_scores) == 0:
        raise ValueError("Cannot summarize an empty collection of fold scores")
    return {
        "mean": float(np.mean(fold_scores)),
        "max": float(np.max(fold_scores)),
        "min": float(np.min(fold_scores)),
    }


class ClassificationMetrics(TypedDict):
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]


class FoldResult(ClassificationMetrics):
    fold: int
    selected_epoch: int
    accuracy: float


class SubjectResult(TypedDict):
    mean: float
    max: float
    min: float
    balanced_accuracy: dict[str, float]
    macro_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    class_labels: list[int]
    folds: list[FoldResult]


def compute_classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> ClassificationMetrics:
    """Binary target order [0, 1]; matrix rows=true, columns=predicted.

    Macro F1 includes both classes and assigns zero to undefined class scores.
    Balanced accuracy follows sklearn: average recall of classes present in y_true.
    """
    if len(labels) == 0 or len(labels) != len(predictions):
        raise ValueError("Labels and predictions must be nonempty and have equal length")
    if not set(labels).union(predictions).issubset({0, 1}):
        raise ValueError("Expected binary target labels 0 and 1")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def summarize_fold_results(folds: list[FoldResult]) -> SubjectResult:
    accuracy = summarize_fold_scores([fold["accuracy"] for fold in folds])
    return {
        "mean": accuracy["mean"],
        "max": accuracy["max"],
        "min": accuracy["min"],
        "balanced_accuracy": summarize_fold_scores([fold["balanced_accuracy"] for fold in folds]),
        "macro_f1": summarize_fold_scores([fold["macro_f1"] for fold in folds]),
        "confusion_matrix": np.sum([fold["confusion_matrix"] for fold in folds], axis=0).tolist(),
        "class_labels": [0, 1],
        "folds": folds,
    }
