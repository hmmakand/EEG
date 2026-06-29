"""Description/metadata based helpers for Braindecode datasets."""

from __future__ import annotations

from typing import Any, cast

from braindecode.datasets import BaseConcatDataset
from omegaconf import DictConfig

from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.splitting.config import section
from eeg_bci.data.splitting.types import SplittableBraindecodeDataset


def split_by_session_description(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
) -> tuple[BraindecodeLikeDataset, BraindecodeLikeDataset]:
    session_cfg = section(split_cfg, "session")
    column = str(session_cfg.get("column", split_cfg.get("column", "session")))
    train_key = str(session_cfg.get("train_key", split_cfg.get("train_key", "0train")))
    test_key = str(
        session_cfg.get(
            "test_key",
            split_cfg.get("test_key", split_cfg.get("eval_key", "1test")),
        )
    )

    splits = description_splits(dataset, column)
    missing_keys = [key for key in (train_key, test_key) if key not in splits]
    if missing_keys:
        available = ", ".join(str(key) for key in sorted(splits, key=sort_description_key))
        missing = ", ".join(missing_keys)
        raise KeyError(
            f"Split keys not found for column {column}: {missing}. "
            f"Available keys: {available}."
        )

    return splits[train_key], splits[test_key]


def description_splits(
    dataset: BraindecodeLikeDataset,
    column: str,
) -> dict[Any, BraindecodeLikeDataset]:
    if not hasattr(dataset, "split"):
        raise TypeError(
            "Description-based splitting requires a Braindecode dataset with split()."
        )
    splittable = cast(SplittableBraindecodeDataset, dataset)
    return splittable.split(column)


def concat_braindecode_sources(
    sources: list[BraindecodeLikeDataset],
) -> BraindecodeLikeDataset:
    datasets: list[Any] = []
    for source in sources:
        source_any = cast(Any, source)
        if hasattr(source_any, "datasets"):
            datasets.extend(source_any.datasets)
        else:
            datasets.append(source)
    return BaseConcatDataset(datasets)


def normalize_description_key(key: Any) -> int | str:
    if hasattr(key, "item"):
        key = key.item()
    try:
        return int(key)
    except (TypeError, ValueError):
        return str(key)


def sort_description_key(key: Any) -> tuple[int, int | str]:
    normalized = normalize_description_key(key)
    if isinstance(normalized, int):
        return (0, normalized)
    return (1, normalized)
