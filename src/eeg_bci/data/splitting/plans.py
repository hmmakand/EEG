"""Common split plan builders."""

from __future__ import annotations

from typing import cast

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset, random_split

from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.splitting.config import (
    session_split_strategy,
    split_lengths,
    validation_shuffle,
    validation_size,
)
from eeg_bci.data.splitting.resampling import make_resampler
from eeg_bci.data.splitting.sources import (
    make_chronological_split_source,
    make_session_split_source,
)
from eeg_bci.data.splitting.strategies import (
    CHRONOLOGICAL_SPLIT_STRATEGIES,
    CROSS_VALIDATION_TEST,
    GRID_SEARCH_TEST,
    SESSION_LEAVE_ONE_SUBJECT_OUT,
    SESSION_SPLIT_STRATEGIES,
    TRAIN_TEST,
    TRAIN_VALID_TEST,
    split_method,
)
from eeg_bci.data.splitting.types import Resampler, SizedDataset, SplitPlan


def make_braindecode_protocol_split(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
    *,
    seed: int,
) -> SplitPlan:
    """Create a split plan from a Braindecode description/session protocol."""

    split_strategy = session_split_strategy(split_cfg)
    if split_strategy not in SESSION_SPLIT_STRATEGIES:
        available = ", ".join(sorted(SESSION_SPLIT_STRATEGIES))
        raise ValueError(
            f"Unsupported session split strategy {split_strategy}. "
            f"Available: {available}."
        )
    source = make_session_split_source(dataset, split_cfg)
    return make_split_plan(
        split_strategy,
        source.train_pool,
        source.test_set,
        split_cfg,
        seed=seed,
    )


def make_chronological_protocol_split(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
    *,
    seed: int,
) -> SplitPlan:
    """Create a chronological split plan for session-less EEG datasets."""

    strategy = str(split_cfg.get("split_strategy", split_cfg.get("strategy")))
    if strategy not in CHRONOLOGICAL_SPLIT_STRATEGIES:
        available = ", ".join(sorted(CHRONOLOGICAL_SPLIT_STRATEGIES))
        raise ValueError(
            f"Unsupported chronological split strategy {strategy}. "
            f"Available: {available}."
        )
    source = make_chronological_split_source(dataset, split_cfg)
    return make_split_plan(
        strategy,
        source.train_pool,
        source.test_set,
        split_cfg,
        seed=seed,
        resampler_use_top_level_defaults=False,
    )


def make_split_plan(
    strategy: str,
    train_pool: Dataset,
    test_set: Dataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    resampler_use_top_level_defaults: bool = True,
) -> SplitPlan:
    """Create a SplitPlan once the outer train/test source is known."""

    method = split_method(strategy)
    if method == TRAIN_TEST:
        return make_train_test_plan(strategy, train_pool, test_set)

    if method == TRAIN_VALID_TEST:
        return make_train_valid_test_plan(
            strategy,
            train_pool,
            test_set,
            split_cfg,
            seed=seed,
        )

    if method == CROSS_VALIDATION_TEST:
        return make_resampled_plan(
            strategy,
            train_pool,
            test_set,
            split_cfg,
            seed=seed,
            resampler_use_top_level_defaults=resampler_use_top_level_defaults,
        )

    if method == GRID_SEARCH_TEST:
        return make_grid_search_plan(
            strategy,
            train_pool,
            test_set,
            split_cfg,
            seed=seed,
            resampler_use_top_level_defaults=resampler_use_top_level_defaults,
        )

    raise ValueError(f"Unsupported split methodology {method}.")


def make_train_test_plan(
    strategy: str,
    train_pool: Dataset,
    test_set: Dataset,
) -> SplitPlan:
    return SplitPlan(
        split_strategy=strategy,
        train_pool=train_pool,
        train_set=train_pool,
        valid_set=None,
        test_set=test_set,
        resampler=None,
    )


def make_train_valid_test_plan(
    strategy: str,
    train_pool: Dataset,
    test_set: Dataset,
    split_cfg: DictConfig,
    *,
    seed: int,
) -> SplitPlan:
    train_set, valid_set = split_train_valid(
        train_pool,
        valid_size=validation_size(split_cfg),
        shuffle=validation_shuffle(split_cfg),
        seed=seed,
    )
    return SplitPlan(
        split_strategy=strategy,
        train_pool=train_pool,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        resampler=None,
    )


def make_resampled_plan(
    strategy: str,
    train_pool: Dataset,
    test_set: Dataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    resampler_use_top_level_defaults: bool = True,
) -> SplitPlan:
    resampler = make_resampler(
        split_cfg,
        seed=seed,
        use_top_level_defaults=resampler_use_top_level_defaults,
    )
    return _make_resampled_plan(strategy, train_pool, test_set, resampler)


def make_grid_search_plan(
    strategy: str,
    train_pool: Dataset,
    test_set: Dataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    resampler_use_top_level_defaults: bool = True,
) -> SplitPlan:
    return make_resampled_plan(
        strategy,
        train_pool,
        test_set,
        split_cfg,
        seed=seed,
        resampler_use_top_level_defaults=resampler_use_top_level_defaults,
    )


def make_loso_plan(
    train_set: Dataset,
    valid_set: Dataset | None,
    test_set: Dataset,
    *,
    strategy: str = SESSION_LEAVE_ONE_SUBJECT_OUT,
) -> SplitPlan:
    return SplitPlan(
        split_strategy=strategy,
        train_pool=train_set,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        resampler=None,
    )


def split_train_valid(
    train_set: Dataset,
    *,
    valid_size: float,
    seed: int,
    shuffle: bool,
) -> tuple[Dataset, Dataset]:
    """Split a training set into inner-training and validation subsets."""

    sized_train_set = cast(SizedDataset, train_set)
    train_len, valid_len = split_lengths(len(sized_train_set), holdout_size=valid_size)

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        inner_train_set, valid_set = random_split(
            train_set,
            [train_len, valid_len],
            generator=generator,
        )
        return inner_train_set, valid_set

    indices = list(range(len(sized_train_set)))
    inner_train_indices = indices[:train_len]
    valid_indices = indices[train_len:]
    return Subset(train_set, inner_train_indices), Subset(train_set, valid_indices)


def _make_resampled_plan(
    strategy: str,
    train_pool: Dataset,
    test_set: Dataset,
    resampler: Resampler,
) -> SplitPlan:
    return SplitPlan(
        split_strategy=strategy,
        train_pool=train_pool,
        train_set=train_pool,
        valid_set=None,
        test_set=test_set,
        resampler=resampler,
    )
