"""Outer train/test split source builders."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset, random_split

from eeg_bci.data.adapters import BraindecodeLikeDataset, TensorDatasetFromBraindecode
from eeg_bci.data.splitting.config import split_lengths
from eeg_bci.data.splitting.description import split_by_session_description
from eeg_bci.data.splitting.types import SplitSource


def make_session_split_source(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
) -> SplitSource:
    """Create the outer train/test source from description session metadata."""

    train_source, test_source = split_by_session_description(dataset, split_cfg)
    return SplitSource(
        train_pool=TensorDatasetFromBraindecode(train_source),
        test_set=TensorDatasetFromBraindecode(test_source),
    )


def make_chronological_split_source(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
) -> SplitSource:
    """Create the outer train/test source from chronological window order."""

    test_size = float(split_cfg.get("test_size", 0.2))
    stratify = bool(split_cfg.get("stratify", True))
    tensor_dataset = TensorDatasetFromBraindecode(dataset)
    train_indices, test_indices = _chronological_split_indices(
        dataset, test_size=test_size, stratify=stratify
    )
    train_indices = _sort_chronological_indices(dataset, train_indices)
    test_indices = _sort_chronological_indices(dataset, test_indices)
    return SplitSource(
        train_pool=Subset(tensor_dataset, train_indices.tolist()),
        test_set=Subset(tensor_dataset, test_indices.tolist()),
    )


def split_by_description(
    dataset: BraindecodeLikeDataset, split_cfg: DictConfig
) -> tuple[Dataset, Dataset]:
    """Split a Braindecode dataset using description metadata."""

    source = make_session_split_source(dataset, split_cfg)
    return source.train_pool, source.test_set


def split_random_train_test(
    dataset: BraindecodeLikeDataset,
    *,
    test_size: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Create a deterministic random train/test split for smoke-test datasets."""

    tensor_dataset = TensorDatasetFromBraindecode(dataset)
    train_len, test_len = split_lengths(len(tensor_dataset), holdout_size=test_size)
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(
        tensor_dataset,
        [train_len, test_len],
        generator=generator,
    )
    return train_set, test_set


def split_chronological_train_test(
    dataset: BraindecodeLikeDataset,
    *,
    test_size: float,
    stratify: bool,
    seed: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Create a chronological train/test split for session-less EEG datasets.

    Windows are ordered by their start position in the original recording
    (``i_start_in_trial``). The earliest windows form the training set and the
    latest windows form the test set. This respects temporal ordering and
    non-stationarity, which is important when no official session split exists.

    When ``stratify`` is true, each class is split independently so that class
    proportions are preserved in both sets.
    """

    _ = seed
    split_cfg = OmegaConf.create({"test_size": test_size, "stratify": stratify})
    source = make_chronological_split_source(dataset, split_cfg)
    return source.train_pool, source.test_set


def _chronological_split_indices(
    dataset: BraindecodeLikeDataset,
    test_size: float,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return chronologically stratified train and test indices."""

    metadata = cast(Any, dataset).get_metadata()
    targets = metadata["target"].to_numpy()

    if stratify:
        train_indices: list[int] = []
        test_indices: list[int] = []
        for label in np.unique(targets):
            class_idx = np.where(targets == label)[0]
            class_idx_sorted = class_idx[
                np.argsort(metadata["i_start_in_trial"].iloc[class_idx].to_numpy())
            ]
            _, n_test_class = split_lengths(
                len(class_idx_sorted), holdout_size=test_size
            )
            train_indices.extend(class_idx_sorted[:-n_test_class].tolist())
            test_indices.extend(class_idx_sorted[-n_test_class:].tolist())
        return np.array(train_indices), np.array(test_indices)

    sorted_idx = metadata["i_start_in_trial"].to_numpy().argsort()
    train_len, _ = split_lengths(len(sorted_idx), holdout_size=test_size)
    return sorted_idx[:train_len], sorted_idx[train_len:]


def _sort_chronological_indices(
    dataset: BraindecodeLikeDataset,
    indices: np.ndarray,
) -> np.ndarray:
    metadata = cast(Any, dataset).get_metadata()
    return indices[np.argsort(metadata["i_start_in_trial"].iloc[indices].to_numpy())]
