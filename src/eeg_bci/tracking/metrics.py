from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

DEFAULT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "cohen_kappa",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "confusion_matrix",
    "roc_auc",
]

SUPPORTED_METRICS = frozenset(
    {
        *DEFAULT_METRICS,
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
    }
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
    metrics: list[str] | None = None,
    labels: list[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a requested set of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    y_prob : np.ndarray | None
        Predicted class probabilities, shape (n_samples, n_classes).
        Required for ``roc_auc``.
    class_names : list[str] | None
        Optional class names for per-class reports.
    metrics : list[str] | None
        Metric names to compute. Defaults to :data:`DEFAULT_METRICS`.
    labels : list[int] | np.ndarray | None
        Explicit label IDs in classifier/probability-column order. Supplying
        this keeps confusion matrices and per-class metrics stable when a
        particular split is missing one or more classes.

    Returns
    -------
    dict[str, Any]
        Mapping from metric name to value. Scalar metrics are floats;
        ``confusion_matrix`` is a nested list; per-class metrics are dicts.
    """
    metrics = normalize_eval_metrics(metrics)
    requested = set(metrics)
    label_array = _label_array(y_true, y_pred, labels)
    result: dict[str, Any] = {}

    if "accuracy" in requested:
        result["accuracy"] = float(accuracy_score(y_true, y_pred))

    if "balanced_accuracy" in requested:
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    if "cohen_kappa" in requested:
        result["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred, labels=label_array))

    if "macro_f1" in requested:
        result["macro_f1"] = float(
            f1_score(y_true, y_pred, labels=label_array, average="macro", zero_division=0)
        )

    if "macro_precision" in requested:
        result["macro_precision"] = float(
            precision_score(
                y_true,
                y_pred,
                labels=label_array,
                average="macro",
                zero_division=0,
            )
        )

    if "macro_recall" in requested:
        result["macro_recall"] = float(
            recall_score(y_true, y_pred, labels=label_array, average="macro", zero_division=0)
        )

    if "per_class_precision" in requested:
        result["per_class_precision"] = _per_class_metric(
            y_true, y_pred, class_names, labels=label_array, metric="precision"
        )

    if "per_class_recall" in requested:
        result["per_class_recall"] = _per_class_metric(
            y_true, y_pred, class_names, labels=label_array, metric="recall"
        )

    if "per_class_f1" in requested:
        result["per_class_f1"] = _per_class_metric(
            y_true, y_pred, class_names, labels=label_array, metric="f1"
        )

    if "confusion_matrix" in requested:
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=label_array).tolist()

    if "roc_auc" in requested:
        roc_value = _compute_roc_auc(y_true, y_prob, labels=label_array)
        result["roc_auc"] = float("nan") if roc_value is None else roc_value

    return result


def normalize_eval_metrics(metrics: Any, *, include_accuracy: bool = True) -> list[str]:
    """Validate and normalize configured evaluation metric names.

    Accuracy is kept as an internal invariant because training code uses it as
    the canonical test score and backward-compatible summary value.
    """
    if metrics is None:
        normalized = list(DEFAULT_METRICS)
    elif isinstance(metrics, str):
        normalized = [metrics]
    elif isinstance(metrics, (list, tuple)):
        normalized = list(metrics)
    else:
        raise TypeError(
            "training.eval_metrics must be a string or a list of strings; "
            f"got {type(metrics).__name__}."
        )

    if not all(isinstance(metric, str) for metric in normalized):
        raise TypeError("training.eval_metrics must contain only strings.")

    unknown = sorted(set(normalized).difference(SUPPORTED_METRICS))
    if unknown:
        available = ", ".join(sorted(SUPPORTED_METRICS))
        requested = ", ".join(unknown)
        raise ValueError(
            f"Unsupported evaluation metric(s): {requested}. Available metrics: {available}."
        )

    deduplicated = list(dict.fromkeys(normalized))
    if include_accuracy and "accuracy" not in deduplicated:
        deduplicated.insert(0, "accuracy")
    return deduplicated


def summarize_scalar_metrics(
    rows: list[dict[str, Any]],
    *,
    metric_prefix: str = "test_",
    weight_key: str = "n_test_windows",
) -> dict[str, float]:
    """Summarize scalar metrics across grouped experiment rows."""
    metric_keys = sorted(
        key
        for key in {key for row in rows for key in row}
        if key.startswith(metric_prefix)
    )
    summary: dict[str, float] = {}
    for key in metric_keys:
        pairs = _numeric_weight_pairs(rows, key=key, weight_key=weight_key)
        if not pairs:
            continue
        values = np.asarray([value for value, _ in pairs], dtype=float)
        weights = np.asarray([weight for _, weight in pairs], dtype=float)
        total_weight = float(weights.sum())
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
        summary[f"{key}_min"] = float(values.min())
        summary[f"{key}_max"] = float(values.max())
        summary[f"{key}_weighted"] = (
            float(np.average(values, weights=weights)) if total_weight > 0.0 else float("nan")
        )
    return summary


def _per_class_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None,
    labels: np.ndarray,
    metric: str,
) -> dict[str, float]:
    names = class_names or [str(label) for label in labels]
    if len(names) != len(labels):
        raise ValueError(
            f"class_names length ({len(names)}) must match labels length ({len(labels)})."
        )
    if metric == "recall":
        raw_values = recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
    elif metric == "precision":
        raw_values = precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
    elif metric == "f1":
        raw_values = f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
    else:
        raise ValueError(f"Unsupported per-class metric: {metric}")
    values = np.asarray(raw_values)
    return {str(name): float(value) for name, value in zip(names, values)}


def _compute_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray | None,
    *,
    labels: np.ndarray,
) -> float | None:
    if y_prob is None:
        logger.warning("roc_auc requested but no predicted probabilities were provided.")
        return None

    n_classes = len(labels)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2 and y_prob.shape[1] >= n_classes:
        y_prob = y_prob[:, labels]

    try:
        if n_classes == 2:
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                return float(roc_auc_score(y_true, y_prob[:, 1]))
            return float(roc_auc_score(y_true, y_prob))
        if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
            logger.warning(
                "roc_auc for multiclass requires probabilities with shape "
                "(n_samples, n_classes); got %s.",
                y_prob.shape,
            )
            return None
        return float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        )
    except ValueError as exc:
        logger.warning("roc_auc could not be computed: %s", exc)
        return None


def _label_array(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | np.ndarray | None,
) -> np.ndarray:
    if labels is not None:
        label_array = np.asarray(labels, dtype=np.int64)
    else:
        label_array = np.asarray(sorted(set(y_true).union(set(y_pred))), dtype=np.int64)
    if label_array.ndim != 1 or len(label_array) == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence.")
    return label_array


def _numeric_weight_pairs(
    rows: list[dict[str, Any]],
    *,
    key: str,
    weight_key: str,
) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    for row in rows:
        value = row.get(key)
        weight = row.get(weight_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            continue
        pairs.append((numeric, int(weight)))
    return pairs
