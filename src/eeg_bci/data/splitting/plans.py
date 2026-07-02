"""Common split plan builders."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset, random_split

from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.splitting.config import (
    grouped_split_indices,
    resolved_source_and_method,
    split_lengths,
    validation_shuffle,
    validation_size,
)
from eeg_bci.data.splitting.resampling import make_resampler
from eeg_bci.data.splitting.sources import build_split_source
from eeg_bci.data.splitting.strategies import (
    CROSS_VALIDATION_TEST,
    GRID_SEARCH_TEST,
    LOSO,
    METHODS,
    SOURCE_CHRONOLOGICAL,
    TRAIN_TEST,
    TRAIN_VALID_TEST,
    split_label,
)
from eeg_bci.data.splitting.types import SizedDataset, SplitPlan


def make_protocol_split(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
    *,
    seed: int,
) -> SplitPlan:
    """Create a split plan for any registered split source/method pair."""

    source, method = resolved_source_and_method(split_cfg)
    if method not in METHODS:
        available = ", ".join(sorted(METHODS))
        raise ValueError(f"Unsupported split method {method!r}. Available: {available}.")
    if method == LOSO:
        raise ValueError(
            f"Split method {LOSO!r} returns multiple folds and must be built with "
            "make_leave_one_subject_out_folds(), not make_protocol_split()."
        )

    split_source = build_split_source(source, dataset, split_cfg, seed=seed)
    return make_split_plan(
        source,
        method,
        split_source.train_pool,
        split_source.test_set,
        split_cfg,
        seed=seed,
        resampler_use_top_level_defaults=(source != SOURCE_CHRONOLOGICAL),
        groups=split_source.groups,
    )


def make_split_plan(
    source: str,
    method: str,
    train_pool: Dataset,
    test_set: Dataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    resampler_use_top_level_defaults: bool = True,
    groups: np.ndarray | None = None,
) -> SplitPlan:
    """Create a SplitPlan once the outer train/test source is known.

    ``groups`` is the per-row subject id for ``train_pool`` (see
    ``SplitSource.groups``). When provided, validation splits and resampling
    folds are computed proportionally per subject instead of as a flat
    positional/shuffled cut across the whole pooled ``train_pool``.
    """

    label = split_label(source, method)
    if method == TRAIN_TEST:
        return make_train_test_plan(label, train_pool, test_set)

    if method == TRAIN_VALID_TEST:
        return make_train_valid_test_plan(
            label,
            train_pool,
            test_set,
            split_cfg,
            seed=seed,
            groups=groups,
        )

    if method == CROSS_VALIDATION_TEST:
        return make_resampled_plan(
            label,
            CROSS_VALIDATION_TEST,
            train_pool,
            test_set,
            split_cfg,
            seed=seed,
            resampler_use_top_level_defaults=resampler_use_top_level_defaults,
            groups=groups,
        )

    if method == GRID_SEARCH_TEST:
        return make_grid_search_plan(
            label,
            train_pool,
            test_set,
            split_cfg,
            seed=seed,
            resampler_use_top_level_defaults=resampler_use_top_level_defaults,
            groups=groups,
        )

    raise ValueError(f"Unsupported split methodology {method}.")


def make_train_test_plan(
    label: str,
    train_pool: Dataset,
    test_set: Dataset,
) -> SplitPlan:
    return SplitPlan(
        split_strategy=label,
        method=TRAIN_TEST,
        train_pool=train_pool,
        train_set=train_pool,
        valid_set=None,
        test_set=test_set,
        resampler=None,
    )


def make_train_valid_test_plan(
    label: str,
    train_pool: Dataset,
    test_set: Dataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    groups: np.ndarray | None = None,
) -> SplitPlan:
    train_set, valid_set = split_train_valid(
        train_pool,
        valid_size=validation_size(split_cfg),
        shuffle=validation_shuffle(split_cfg),
        seed=seed,
        groups=groups,
    )
    return SplitPlan(
        split_strategy=label,
        method=TRAIN_VALID_TEST,
        train_pool=train_pool,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        resampler=None,
    )


def make_resampled_plan(
    label: str,
    method: str,
    train_pool: Dataset,
    test_set: Dataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    resampler_use_top_level_defaults: bool = True,
    groups: np.ndarray | None = None,
) -> SplitPlan:
    resampler = make_resampler(
        split_cfg,
        seed=seed,
        use_top_level_defaults=resampler_use_top_level_defaults,
        groups=groups,
    )
    return SplitPlan(
        split_strategy=label,
        method=method,
        train_pool=train_pool,
        train_set=train_pool,
        valid_set=None,
        test_set=test_set,
        resampler=resampler,
    )


def make_grid_search_plan(
    label: str,
    train_pool: Dataset,
    test_set: Dataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    resampler_use_top_level_defaults: bool = True,
    groups: np.ndarray | None = None,
) -> SplitPlan:
    return make_resampled_plan(
        label,
        GRID_SEARCH_TEST,
        train_pool,
        test_set,
        split_cfg,
        seed=seed,
        resampler_use_top_level_defaults=resampler_use_top_level_defaults,
        groups=groups,
    )


def make_loso_plan(
    train_set: Dataset,
    valid_set: Dataset | None,
    test_set: Dataset,
    *,
    source: str,
) -> SplitPlan:
    return SplitPlan(
        split_strategy=split_label(source, LOSO),
        method=LOSO,
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
    groups: np.ndarray | None = None,
) -> tuple[Dataset, Dataset]:
    """Split a training set into inner-training and validation subsets.

    When ``groups`` is given (e.g. a per-row subject id), the split is
    computed independently within each group and unioned, so every group is
    proportionally represented in both the inner-training and validation
    sets instead of a flat cut that can land entirely within one group when
    rows are ordered group-by-group (as pooled multi-subject data is).
    """

    sized_train_set = cast(SizedDataset, train_set)
    n = len(sized_train_set)

    if groups is not None:
        inner_train_indices, valid_indices = grouped_split_indices(
            n, groups, holdout_size=valid_size, shuffle=shuffle, seed=seed
        )
        return (
            Subset(train_set, inner_train_indices.tolist()),
            Subset(train_set, valid_indices.tolist()),
        )

    train_len, valid_len = split_lengths(n, holdout_size=valid_size)

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        inner_train_set, valid_set = random_split(
            train_set,
            [train_len, valid_len],
            generator=generator,
        )
        return inner_train_set, valid_set

    indices = list(range(n))
    inner_train_indices = indices[:train_len]
    valid_indices = indices[train_len:]
    return Subset(train_set, inner_train_indices), Subset(train_set, valid_indices)
