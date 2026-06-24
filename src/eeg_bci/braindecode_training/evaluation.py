from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
from torch.utils.data import Dataset, Subset


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
    y = collect_targets(dataset)
    return float(classifier.score(dataset, y=y))


def score_classifier_by_description(
    classifier: Any,
    dataset: Dataset,
    *,
    description_key: str,
) -> list[dict[str, int | float | str]]:
    """Score a classifier on groups defined by Braindecode descriptions."""

    groups = _indices_by_description(dataset, description_key)
    rows: list[dict[str, int | float | str]] = []
    for group_name, indices in sorted(groups.items(), key=_sort_group_item):
        group_dataset = Subset(dataset, indices)
        rows.append(
            {
                description_key: group_name,
                "n_windows": len(indices),
                "test_acc": score_classifier(classifier, group_dataset),
            }
        )
    return rows


def latest_history_value(classifier: Any, key: str, default: float = 0.0) -> float:
    try:
        value = classifier.history[-1, key]
    except (IndexError, KeyError):
        return default
    return float(value)


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
