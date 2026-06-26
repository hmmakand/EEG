from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
from torch.utils.data import Dataset, Subset

from eeg_bci.tracking.metrics import compute_metrics


class IndexedDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


def collect_targets(dataset: Dataset) -> np.ndarray:
    indexed_dataset = cast(IndexedDataset, dataset)
    targets: list[int] = []
    for index in range(len(indexed_dataset)):
        sample = indexed_dataset[index]
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise TypeError("Expected dataset samples to contain at least (x, y).")
        targets.append(_as_int(sample[1]))
    return np.asarray(targets, dtype=np.int64)


def score_classifier(classifier: Any, dataset: Dataset) -> float:
    """Return overall classification accuracy (legacy helper)."""
    y = collect_targets(dataset)
    return float(classifier.score(dataset, y=y))


def evaluate_classifier(
    classifier: Any,
    dataset: Dataset,
    *,
    metrics: list[str] | None = None,
    class_names: list[str] | None = None,
    labels: list[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a configurable set of metrics on a trained classifier.

    Parameters
    ----------
    classifier : Any
        A trained ``EEGClassifier`` or any object with ``predict`` and
        optionally ``predict_proba`` methods.
    dataset : Dataset
        Dataset yielding ``(x, y)`` samples.
    metrics : list[str] | None
        Metric names to compute. Defaults to the package defaults. Accuracy is
        always included by the metrics layer for internal score stability.
    class_names : list[str] | None
        Optional class names for per-class reports and plots.
    labels : list[int] | np.ndarray | None
        Explicit label IDs in classifier/probability-column order.

    Returns
    -------
    dict[str, Any]
        Computed metrics. Always contains ``accuracy``.
    """
    y_true = collect_targets(dataset)
    y_pred = classifier.predict(dataset)
    y_prob = _predict_proba(classifier, dataset)
    return compute_metrics(
        y_true,
        y_pred,
        y_prob=y_prob,
        class_names=class_names,
        metrics=metrics,
        labels=labels,
    )


def score_classifier_by_description(
    classifier: Any,
    dataset: Dataset,
    *,
    description_key: str,
    metrics: list[str] | None = None,
    class_names: list[str] | None = None,
    labels: list[int] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Score a classifier on groups defined by Braindecode descriptions."""

    groups = _indices_by_description(dataset, description_key)
    rows: list[dict[str, Any]] = []
    for group_name, indices in sorted(groups.items(), key=_sort_group_item):
        group_dataset = Subset(dataset, indices)
        group_metrics = evaluate_classifier(
            classifier,
            group_dataset,
            metrics=metrics,
            class_names=class_names,
            labels=labels,
        )
        rows.append(
            {
                description_key: group_name,
                "n_windows": len(indices),
                **group_metrics,
            }
        )
    return rows


def latest_history_value(classifier: Any, key: str, default: float = 0.0) -> float:
    try:
        value = classifier.history[-1, key]
    except (IndexError, KeyError):
        return default
    return float(value)


def _predict_proba(classifier: Any, dataset: Dataset) -> np.ndarray | None:
    if not hasattr(classifier, "predict_proba"):
        return None
    try:
        return np.asarray(classifier.predict_proba(dataset))
    except Exception:  # noqa: BLE001
        return None


def _as_int(value: Any) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _indices_by_description(dataset: Dataset, description_key: str) -> dict[str, list[int]]:
    source = _braindecode_source(dataset)
    if source is None or not hasattr(source, "datasets"):
        return {}

    groups: dict[str, list[int]] = {}
    offset = 0
    for index, sub_dataset in enumerate(source.datasets):
        value = _description_value(source, sub_dataset, index, description_key)
        next_offset = offset + len(sub_dataset)
        if value is not None:
            groups.setdefault(str(value), []).extend(range(offset, next_offset))
        offset = next_offset
    return groups


def _braindecode_source(dataset: Dataset) -> Any | None:
    source = getattr(dataset, "dataset", None)
    if source is not None and hasattr(source, "datasets"):
        return source
    if hasattr(dataset, "datasets"):
        return dataset
    return None


def _description_value(
    source: Any,
    sub_dataset: Any,
    index: int,
    description_key: str,
) -> Any | None:
    value = _get_description_value(
        getattr(sub_dataset, "description", None),
        description_key,
    )
    if value is not None:
        return value

    description = getattr(source, "description", None)
    if description is None:
        return None
    if hasattr(description, "iloc"):
        value = _get_description_value(description.iloc[index], description_key)
        if value is not None:
            return value
    return _get_description_value(description, description_key)


def _get_description_value(description: Any, key: str) -> Any | None:
    if description is None:
        return None
    if hasattr(description, "get"):
        value = description.get(key, None)
    else:
        value = getattr(description, key, None)
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _sort_group_item(item: tuple[str, list[int]]) -> tuple[int, int | str]:
    group_name, _ = item
    try:
        return (0, int(group_name))
    except ValueError:
        return (1, group_name)
