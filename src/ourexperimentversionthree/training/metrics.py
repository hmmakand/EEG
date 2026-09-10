"""Classification metrics for graph-level broadcast-11 predictions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregate metrics for one complete loader pass.

    Mirrors the manuscript's reported metric set (Accuracy, F1, Recall,
    Precision, AUC). F1/Recall/Precision are binary metrics for the positive
    class (label 1, ``right_hand``) rather than macro-averaged, matching how
    the manuscript reports a single value per configuration.
    """

    loss: float
    accuracy: float
    f1: float
    recall: float
    precision: float
    auc: float
    examples: int

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def calculate_metrics(
    *,
    total_loss: float,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> ClassificationMetrics:
    """Calculate aggregate classification metrics from complete predictions.

    ``scores`` is the predicted probability of the positive class (label 1)
    for every example, used only for AUC; ``predictions`` are the hard
    argmax class predictions used for accuracy/F1/recall/precision.
    """

    targets = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    if targets.ndim != 1 or predicted.shape != targets.shape or len(targets) == 0:
        raise ValueError(
            "labels and predictions must be equally sized non-empty vectors"
        )
    if score_values.shape != targets.shape:
        raise ValueError("scores must be the same shape as labels")
    if not math.isfinite(total_loss):
        raise ValueError("total_loss must be finite")
    if len(np.unique(targets)) < 2:
        raise ValueError("AUC requires both classes to be present in the evaluated set")
    return ClassificationMetrics(
        loss=float(total_loss / len(targets)),
        accuracy=float(np.mean(targets == predicted)),
        f1=float(f1_score(targets, predicted, pos_label=1, zero_division=0)),
        recall=float(recall_score(targets, predicted, pos_label=1, zero_division=0)),
        precision=float(
            precision_score(targets, predicted, pos_label=1, zero_division=0)
        ),
        auc=float(roc_auc_score(targets, score_values)),
        examples=len(targets),
    )
