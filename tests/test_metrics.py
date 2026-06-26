from __future__ import annotations

import math

import numpy as np
import pytest

from eeg_bci.tracking.metrics import compute_metrics, normalize_eval_metrics, summarize_scalar_metrics


def test_normalize_eval_metrics_keeps_accuracy_for_internal_score() -> None:
    assert normalize_eval_metrics(["balanced_accuracy"]) == ["accuracy", "balanced_accuracy"]


def test_normalize_eval_metrics_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unsupported evaluation metric"):
        normalize_eval_metrics(["accuracy", "macro_precisionn"])


def test_compute_metrics_uses_explicit_label_space_for_missing_class() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    y_pred = np.asarray([0, 0, 1, 1])
    y_prob = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
        y_prob=y_prob,
        class_names=["zero", "one", "two"],
        labels=[0, 1, 2],
        metrics=["confusion_matrix", "macro_recall", "per_class_recall", "roc_auc"],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 0]]
    assert metrics["macro_recall"] == pytest.approx(2 / 3)
    assert metrics["per_class_recall"] == {"zero": 1.0, "one": 1.0, "two": 0.0}
    assert math.isnan(metrics["roc_auc"])


def test_summarize_scalar_metrics_pairs_values_with_matching_weights() -> None:
    rows = [
        {"n_test_windows": 10, "test_accuracy": 0.1},
        {"n_test_windows": 100},
        {"n_test_windows": 30, "test_accuracy": 0.9, "test_roc_auc": 0.8},
    ]

    summary = summarize_scalar_metrics(rows)

    assert summary["test_accuracy_mean"] == pytest.approx(0.5)
    assert summary["test_accuracy_weighted"] == pytest.approx(0.7)
    assert summary["test_accuracy_min"] == pytest.approx(0.1)
    assert summary["test_accuracy_max"] == pytest.approx(0.9)
    assert summary["test_roc_auc_mean"] == pytest.approx(0.8)
