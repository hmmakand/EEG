"""Inner validation and cross-validation resamplers."""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import BaseCrossValidator, KFold

from eeg_bci.data.splitting.config import section, split_lengths, validation_size
from eeg_bci.data.splitting.types import Resampler


class HoldoutSplit(BaseCrossValidator):
    """One deterministic train/validation split for sklearn CV APIs."""

    def __init__(self, *, valid_size: float, shuffle: bool, seed: int) -> None:
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.seed = seed

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return 1

    def split(self, X: Any, y: Any = None, groups: Any = None):
        train_len, _ = split_lengths(len(X), holdout_size=self.valid_size)
        indices = np.arange(len(X))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        yield indices[:train_len], indices[train_len:]


def make_resampler(
    split_cfg: DictConfig,
    *,
    seed: int,
    default_shuffle: bool = False,
    use_top_level_defaults: bool = True,
) -> Resampler:
    """Create the resampler used inside the training pool."""

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
        random_state = seed if shuffle else None
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    if method == "holdout":
        valid_size = float(resampling_cfg.get("valid_size", validation_size(split_cfg)))
        return HoldoutSplit(valid_size=valid_size, shuffle=shuffle, seed=seed)

    raise ValueError(f"Unsupported resampling method {method}.")


def make_chronological_resampler(split_cfg: DictConfig, *, seed: int) -> Resampler:
    """Create a chronological resampler for session-less datasets.

    Chronological resamplers always default to ``shuffle=false`` so that folds
    are consecutive blocks in time, matching the temporal ordering of the
    training pool.
    """

    return make_resampler(
        split_cfg,
        seed=seed,
        default_shuffle=False,
        use_top_level_defaults=False,
    )
