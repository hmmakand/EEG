from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
from torch.utils.data import Dataset


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
