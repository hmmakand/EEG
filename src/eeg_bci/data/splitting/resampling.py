"""Inner validation and cross-validation resamplers."""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import BaseCrossValidator, KFold

from eeg_bci.data.splitting.config import (
    grouped_split_indices,
    section,
    split_lengths,
    validation_size,
)
from eeg_bci.data.splitting.types import Resampler


class HoldoutSplit(BaseCrossValidator):
    """One deterministic train/validation split for sklearn CV APIs."""

    def __init__(
        self,
        *,
        valid_size: float,
        shuffle: bool,
        seed: int,
        groups: np.ndarray | None = None,
    ) -> None:
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.seed = seed
        self.groups = groups

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return 1

    def split(self, X: Any, y: Any = None, groups: Any = None):
        if self.groups is not None:
            train_idx, valid_idx = grouped_split_indices(
                len(X),
                self.groups,
                holdout_size=self.valid_size,
                shuffle=self.shuffle,
                seed=self.seed,
            )
            yield train_idx, valid_idx
            return

        train_len, _ = split_lengths(len(X), holdout_size=self.valid_size)
        indices = np.arange(len(X))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        yield indices[:train_len], indices[train_len:]


class PerGroupKFold(BaseCrossValidator):
    """K-fold splitter computed independently within each group and unioned.

    This is the opposite of sklearn's ``GroupKFold``, which keeps each
    group's rows entirely within one fold (useful for leave-one-subject-out
    style evaluation). Here every group (e.g. subject) is split into
    ``n_splits`` folds on its own, and fold ``k`` across all groups is
    unioned -- so every fold's validation portion is proportionally
    representative of every group, instead of being dominated by whichever
    group happens to occupy that position in the data.
    """

    def __init__(
        self,
        *,
        n_splits: int,
        shuffle: bool,
        seed: int,
        groups: np.ndarray,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        _validate_group_sizes(groups, n_splits)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.groups = groups

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: Any = None):
        n = len(self.groups)
        fold_assignment = np.empty(n, dtype=int)
        for group_value in np.unique(self.groups):
            group_idx = np.where(self.groups == group_value)[0]
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                rng.shuffle(group_idx)
            fold_sizes = np.full(self.n_splits, len(group_idx) // self.n_splits, dtype=int)
            fold_sizes[: len(group_idx) % self.n_splits] += 1
            start = 0
            for fold_idx, size in enumerate(fold_sizes):
                fold_assignment[group_idx[start : start + size]] = fold_idx
                start += size

        for fold_idx in range(self.n_splits):
            valid_mask = fold_assignment == fold_idx
            yield np.where(~valid_mask)[0], np.where(valid_mask)[0]


def _validate_group_sizes(groups: np.ndarray, n_splits: int) -> None:
    unique, counts = np.unique(groups, return_counts=True)
    too_small = unique[counts < n_splits]
    if len(too_small):
        raise ValueError(
            f"Cannot create {n_splits} folds: group(s) {too_small.tolist()} have fewer "
            f"than {n_splits} samples. Reduce resampling.n_splits or exclude these groups."
        )


def make_resampler(
    split_cfg: DictConfig,
    *,
    seed: int,
    default_shuffle: bool = False,
    use_top_level_defaults: bool = True,
    groups: np.ndarray | None = None,
) -> Resampler:
    """Create the resampler used inside the training pool.

    When ``groups`` is given (e.g. a per-row subject id), the resampler
    distributes every group proportionally across every fold/holdout
    partition instead of taking a flat positional or globally-shuffled cut
    that can land entirely within one group.
    """

    resampling_cfg = section(split_cfg, "resampling")
    shuffle_default = (
        bool(split_cfg.get("shuffle", default_shuffle))
        if use_top_level_defaults
        else default_shuffle
    )
    method = str(resampling_cfg.get("method", "kfold"))
    shuffle = bool(resampling_cfg.get("shuffle", shuffle_default))

    if method == "kfold":
        n_splits_default = (
            int(split_cfg.get("n_splits", 5)) if use_top_level_defaults else 5
        )
        n_splits = int(resampling_cfg.get("n_splits", n_splits_default))
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if groups is not None:
            return PerGroupKFold(n_splits=n_splits, shuffle=shuffle, seed=seed, groups=groups)
        random_state = seed if shuffle else None
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    if method == "holdout":
        valid_size = float(resampling_cfg.get("valid_size", validation_size(split_cfg)))
        return HoldoutSplit(valid_size=valid_size, shuffle=shuffle, seed=seed, groups=groups)

    raise ValueError(f"Unsupported resampling method {method}.")


def make_chronological_resampler(
    split_cfg: DictConfig, *, seed: int, groups: np.ndarray | None = None
) -> Resampler:
    """Create a chronological resampler for session-less datasets.

    Chronological resamplers always default to ``shuffle=false`` so that folds
    are consecutive blocks in time, matching the temporal ordering of the
    training pool.
    """

    return make_resampler(
        split_cfg,
        seed=seed,
        use_top_level_defaults=False,
        default_shuffle=False,
        groups=groups,
    )
