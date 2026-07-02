"""Outer train/test split source builders.

Adding a new dataset-structure-dependent split mechanism only requires
writing one ``build_xxx_source`` function with the signature below and
registering it in ``SOURCE_BUILDERS`` -- no changes are needed in
``plans.py``, ``loso.py``, or ``trainer.py``, since those only deal with the
dataset-agnostic split *method* applied on top of the resulting
``SplitSource``.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset, random_split

from eeg_bci.data.adapters import BraindecodeLikeDataset, TensorDatasetFromBraindecode
from eeg_bci.data.splitting.config import resolve_subject_column, split_lengths
from eeg_bci.data.splitting.description import split_by_session_description
from eeg_bci.data.splitting.strategies import (
    SOURCE_CHRONOLOGICAL,
    SOURCE_RANDOM,
    SOURCE_SESSION,
)
from eeg_bci.data.splitting.types import SplitSource


def make_session_split_source(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
) -> SplitSource:
    """Create the outer train/test source from description session metadata."""

    train_source, test_source = split_by_session_description(dataset, split_cfg)
    subject_column = resolve_subject_column(split_cfg)
    groups = _safe_group_values(cast(Any, train_source).get_metadata(), subject_column)
    return SplitSource(
        train_pool=TensorDatasetFromBraindecode(train_source),
        test_set=TensorDatasetFromBraindecode(test_source),
        groups=groups,
    )


def make_chronological_split_source(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
) -> SplitSource:
    """Create the outer train/test source from chronological window order.

    The split is computed independently per subject (see
    ``_chronological_split_indices``) so that pooling multiple subjects does
    not mix their unrelated, recording-local time axes together.
    """

    test_size = float(split_cfg.get("test_size", 0.2))
    stratify = bool(split_cfg.get("stratify", True))
    subject_column = resolve_subject_column(split_cfg)
    tensor_dataset = TensorDatasetFromBraindecode(dataset)
    train_indices, test_indices = _chronological_split_indices(
        dataset, test_size=test_size, stratify=stratify, subject_column=subject_column
    )
    train_indices = _sort_chronological_indices(dataset, train_indices)
    test_indices = _sort_chronological_indices(dataset, test_indices)

    metadata = cast(Any, dataset).get_metadata()
    full_groups = _safe_group_values(metadata, subject_column)
    groups = full_groups[train_indices] if full_groups is not None else None
    return SplitSource(
        train_pool=Subset(tensor_dataset, train_indices.tolist()),
        test_set=Subset(tensor_dataset, test_indices.tolist()),
        groups=groups,
    )


def _safe_group_values(metadata: Any, subject_column: str) -> np.ndarray | None:
    if subject_column not in metadata.columns:
        return None
    return metadata[subject_column].to_numpy()


def build_random_source(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig | None,
    *,
    seed: int,
) -> SplitSource:
    """Create the outer train/test source from a deterministic random split."""

    test_size = float(split_cfg.get("test_size", 0.2)) if split_cfg is not None else 0.2
    train_set, test_set = split_random_train_test(dataset, test_size=test_size, seed=seed)
    return SplitSource(train_pool=train_set, test_set=test_set)


class SourceBuilder(Protocol):
    def __call__(
        self,
        dataset: BraindecodeLikeDataset,
        split_cfg: DictConfig | None,
        *,
        seed: int,
    ) -> SplitSource: ...


def _require_split_cfg(split_cfg: DictConfig | None, source: str) -> DictConfig:
    if split_cfg is None:
        raise ValueError(f"dataset.split must be configured to use source={source!r}.")
    return split_cfg


def _build_session_source(
    dataset: BraindecodeLikeDataset, split_cfg: DictConfig | None, *, seed: int
) -> SplitSource:
    del seed
    return make_session_split_source(dataset, _require_split_cfg(split_cfg, SOURCE_SESSION))


def _build_chronological_source(
    dataset: BraindecodeLikeDataset, split_cfg: DictConfig | None, *, seed: int
) -> SplitSource:
    del seed
    return make_chronological_split_source(
        dataset, _require_split_cfg(split_cfg, SOURCE_CHRONOLOGICAL)
    )


SOURCE_BUILDERS: dict[str, SourceBuilder] = {
    SOURCE_SESSION: _build_session_source,
    SOURCE_CHRONOLOGICAL: _build_chronological_source,
    SOURCE_RANDOM: build_random_source,
}


def build_split_source(
    source: str,
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig | None,
    *,
    seed: int,
) -> SplitSource:
    """Build the outer train/test source for a registered split source name."""

    try:
        builder = SOURCE_BUILDERS[source]
    except KeyError as exc:
        available = ", ".join(sorted(SOURCE_BUILDERS))
        raise ValueError(f"Unsupported split source {source!r}. Available: {available}.") from exc
    return builder(dataset, split_cfg, seed=seed)


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
    subject_column: str = "subject",
) -> tuple[np.ndarray, np.ndarray]:
    """Return chronologically stratified train and test indices.

    ``i_start_in_trial`` is a sample offset relative to each subject's own
    recording, not a timeline shared across subjects. Pooling multiple
    subjects and sorting/cutting on this field globally would interleave
    unrelated recordings as if they happened on one shared clock. To avoid
    that, the chronological cutoff is computed independently per subject (and
    per class, when ``stratify`` is true) and the resulting indices are
    unioned. A dataset with a single subject, or no subject column at all,
    reduces to one group and behaves exactly as a global split would.
    """

    metadata = cast(Any, dataset).get_metadata()
    targets = metadata["target"].to_numpy()
    subject_values = (
        metadata[subject_column].to_numpy()
        if subject_column in metadata.columns
        else np.zeros(len(metadata), dtype=int)
    )

    train_indices: list[int] = []
    test_indices: list[int] = []
    for subject_value in np.unique(subject_values):
        subject_idx = np.where(subject_values == subject_value)[0]

        if stratify:
            for label in np.unique(targets[subject_idx]):
                class_idx = subject_idx[targets[subject_idx] == label]
                class_idx_sorted = class_idx[
                    np.argsort(metadata["i_start_in_trial"].iloc[class_idx].to_numpy())
                ]
                _, n_test_class = split_lengths(
                    len(class_idx_sorted), holdout_size=test_size
                )
                train_indices.extend(class_idx_sorted[:-n_test_class].tolist())
                test_indices.extend(class_idx_sorted[-n_test_class:].tolist())
            continue

        subject_idx_sorted = subject_idx[
            np.argsort(metadata["i_start_in_trial"].iloc[subject_idx].to_numpy())
        ]
        train_len, _ = split_lengths(len(subject_idx_sorted), holdout_size=test_size)
        train_indices.extend(subject_idx_sorted[:train_len].tolist())
        test_indices.extend(subject_idx_sorted[train_len:].tolist())

    return np.array(train_indices), np.array(test_indices)


def _sort_chronological_indices(
    dataset: BraindecodeLikeDataset,
    indices: np.ndarray,
) -> np.ndarray:
    metadata = cast(Any, dataset).get_metadata()
    return indices[np.argsort(metadata["i_start_in_trial"].iloc[indices].to_numpy())]
