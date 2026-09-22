"""Leave-one-subject-out split construction and shared loader types.

Combination-agnostic: everything here operates only on subject-ID arrays and
generic dataclasses, never on a specific node/edge/band choice. Every
combination module (``without_csd_alpha_wpli.py``, ``csd_alpha_wpli.py``,
...) builds its combination-specific ``PyG`` ``Dataset``/``DataLoader``s on
top of :func:`create_loso_splits` and these types, rather than each
reimplementing subject-disjoint splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch_geometric.loader import DataLoader


@dataclass(frozen=True)
class GraphDataLoaderConfig:
    """Runtime and validation-split options for one LOSO fold."""

    batch_size: int = 32
    validation_subjects: int = 5
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    seed: int = 42
    normalization_epsilon: float = 1e-8
    node_normalization: str = "none"

    def __post_init__(self) -> None:
        if self.node_normalization not in ("none", "zscore"):
            raise ValueError("node_normalization must be none or zscore")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.validation_subjects <= 0:
            raise ValueError("validation_subjects must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if not np.isfinite(self.normalization_epsilon) or self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")


@dataclass(frozen=True)
class LosoGraphSplit:
    """Graph indices and subject IDs assigned to one LOSO fold."""

    train_graph_indices: np.ndarray
    validation_graph_indices: np.ndarray
    test_graph_indices: np.ndarray
    train_subject_ids: tuple[int, ...]
    validation_subject_ids: tuple[int, ...]
    test_subject_id: int
    fold: int


@dataclass(frozen=True)
class FeatureNormalization:
    """Optional training-only statistics, shared identically across subject keys.

    Disabled mode carries empty dictionaries and does not fit any values.
    Training indices record the fitting scope for enabled normalization.
    """

    mean_by_subject: dict[int, torch.Tensor]
    standard_deviation_by_subject: dict[int, torch.Tensor]
    mode: str = "zscore"
    fit_graph_indices: tuple[int, ...] = ()
    epsilon: float = 1e-8


@dataclass(frozen=True)
class LosoDataLoaderBundle:
    """Loaders plus the split and preprocessing provenance for one fold."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    split: LosoGraphSplit
    normalization: FeatureNormalization


@dataclass(frozen=True)
class InnerCvFold:
    """One grouped train/validation split of a development subject set.

    Unlike :class:`LosoGraphSplit`, there is no held-out test subject here --
    inner-CV folds exist purely to score hyperparameter candidates against
    each other using only the outer fold's development (non-test) subjects.
    """

    train_graph_indices: np.ndarray
    validation_graph_indices: np.ndarray
    train_subject_ids: tuple[int, ...]
    validation_subject_ids: tuple[int, ...]
    fold: int


def _graph_indices_for_subjects(
    subject_ids: np.ndarray, selected_subjects: tuple[int, ...]
) -> np.ndarray:
    return np.flatnonzero(np.isin(subject_ids, selected_subjects)).astype(
        np.int64, copy=False
    )


def _validate_split(split: LosoGraphSplit, subject_ids: np.ndarray) -> None:
    graph_sets = (
        set(map(int, split.train_graph_indices)),
        set(map(int, split.validation_graph_indices)),
        set(map(int, split.test_graph_indices)),
    )
    if any(
        graph_sets[left] & graph_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError("LOSO partitions contain overlapping graph indices")
    if set.union(*graph_sets) != set(range(len(subject_ids))):
        raise RuntimeError("LOSO partitions do not cover every graph exactly once")

    subject_sets = (
        set(split.train_subject_ids),
        set(split.validation_subject_ids),
        {split.test_subject_id},
    )
    if any(
        subject_sets[left] & subject_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError("LOSO partitions leak subjects")

    observed_subject_sets = tuple(
        set(map(int, np.unique(subject_ids[indices])))
        for indices in (
            split.train_graph_indices,
            split.validation_graph_indices,
            split.test_graph_indices,
        )
    )
    if observed_subject_sets != subject_sets:
        raise RuntimeError("LOSO graph indices do not match their assigned subjects")


def create_loso_splits(
    subject_ids: np.ndarray,
    *,
    validation_subjects: int = 5,
    seed: int = 42,
) -> tuple[LosoGraphSplit, ...]:
    """Create one reproducible, subject-disjoint fold per observed subject."""

    raw_values = np.asarray(subject_ids)
    if raw_values.ndim != 1 or len(raw_values) == 0:
        raise ValueError("subject_ids must be a non-empty one-dimensional array")
    if not np.issubdtype(raw_values.dtype, np.integer):
        raise TypeError("subject_ids must have an integer dtype")
    values = raw_values.astype(np.int64, copy=False)
    subjects = np.unique(values)
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for train/validation/test")
    if not 1 <= validation_subjects <= len(subjects) - 2:
        raise ValueError(
            "validation_subjects must leave one test and at least one training subject"
        )

    splits: list[LosoGraphSplit] = []
    for fold, test_subject in enumerate(subjects):
        development_subjects = subjects[subjects != test_subject]
        # Including the held-out ID makes each validation draw deterministic but
        # independent across LOSO folds.
        shuffled = np.random.default_rng(seed + int(test_subject)).permutation(
            development_subjects
        )
        validation_ids = tuple(sorted(map(int, shuffled[:validation_subjects])))
        validation_set = set(validation_ids)
        train_ids = tuple(
            sorted(
                int(subject)
                for subject in development_subjects
                if int(subject) not in validation_set
            )
        )
        split = LosoGraphSplit(
            train_graph_indices=_graph_indices_for_subjects(values, train_ids),
            validation_graph_indices=_graph_indices_for_subjects(values, validation_ids),
            test_graph_indices=_graph_indices_for_subjects(
                values, (int(test_subject),)
            ),
            train_subject_ids=train_ids,
            validation_subject_ids=validation_ids,
            test_subject_id=int(test_subject),
            fold=fold,
        )
        _validate_split(split, values)
        splits.append(split)

    return tuple(splits)


def _validate_inner_folds(
    folds: tuple[InnerCvFold, ...],
    subject_ids: np.ndarray,
    eligible_subjects: tuple[int, ...],
) -> None:
    eligible_set = set(int(subject) for subject in eligible_subjects)

    seen_in_validation: set[int] = set()
    for fold in folds:
        validation_set = set(fold.validation_subject_ids)
        if seen_in_validation & validation_set:
            raise RuntimeError(
                "Inner-CV folds assign a subject to validation more than once"
            )
        seen_in_validation |= validation_set
    if seen_in_validation != eligible_set:
        raise RuntimeError(
            "Inner-CV validation folds do not cover every eligible subject exactly once"
        )

    for fold in folds:
        train_set = set(fold.train_subject_ids)
        validation_set = set(fold.validation_subject_ids)
        if train_set & validation_set:
            raise RuntimeError(
                f"Inner-CV fold {fold.fold} leaks subjects between train and validation"
            )
        if train_set | validation_set != eligible_set:
            raise RuntimeError(
                f"Inner-CV fold {fold.fold} does not cover every eligible subject"
            )

        train_graph_set = set(map(int, fold.train_graph_indices))
        validation_graph_set = set(map(int, fold.validation_graph_indices))
        if train_graph_set & validation_graph_set:
            raise RuntimeError(
                f"Inner-CV fold {fold.fold} has overlapping graph indices"
            )

        observed_train = set(map(int, np.unique(subject_ids[fold.train_graph_indices])))
        observed_validation = set(
            map(int, np.unique(subject_ids[fold.validation_graph_indices]))
        )
        if observed_train != train_set or observed_validation != validation_set:
            raise RuntimeError(
                f"Inner-CV fold {fold.fold} graph indices do not match its assigned subjects"
            )


def create_inner_cv_folds(
    subject_ids: np.ndarray,
    eligible_subjects: tuple[int, ...],
    *,
    k: int = 10,
    seed: int = 42,
) -> tuple[InnerCvFold, ...]:
    """Split ``eligible_subjects`` into ``k`` grouped train/validation folds.

    Every subject in ``eligible_subjects`` is assigned whole to exactly one
    fold's validation set across the ``k`` folds (the rest of
    ``eligible_subjects`` form that fold's training set); subjects outside
    ``eligible_subjects`` (e.g. the outer LOSO test subject) never appear in
    any fold. Uses :class:`sklearn.model_selection.GroupKFold` with
    ``shuffle=True, random_state=seed`` for the actual size-balanced,
    seed-controlled grouping (e.g. 49 subjects over 10 folds always yields 9
    folds of 5 and 1 fold of 4, since ``shuffle`` permutes *which* subjects
    land together without changing that size split).
    """

    raw_values = np.asarray(subject_ids)
    if raw_values.ndim != 1 or len(raw_values) == 0:
        raise ValueError("subject_ids must be a non-empty one-dimensional array")
    if not np.issubdtype(raw_values.dtype, np.integer):
        raise TypeError("subject_ids must have an integer dtype")
    values = raw_values.astype(np.int64, copy=False)

    eligible = tuple(int(subject) for subject in eligible_subjects)
    if len(set(eligible)) != len(eligible):
        raise ValueError("eligible_subjects must not contain duplicates")
    if k < 2:
        raise ValueError("k must be at least 2")
    if len(eligible) < k:
        raise ValueError(
            f"k={k} folds requires at least {k} eligible subjects; got {len(eligible)}"
        )
    missing = set(eligible) - set(map(int, np.unique(values)))
    if missing:
        raise ValueError(
            f"eligible_subjects contains subjects absent from subject_ids: {sorted(missing)}"
        )

    eligible_graph_indices = _graph_indices_for_subjects(values, eligible)
    eligible_groups = values[eligible_graph_indices]

    group_k_fold = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds: list[InnerCvFold] = []
    for fold, (train_positions, validation_positions) in enumerate(
        group_k_fold.split(np.zeros(len(eligible_groups)), groups=eligible_groups)
    ):
        train_graph_indices = np.sort(eligible_graph_indices[train_positions])
        validation_graph_indices = np.sort(eligible_graph_indices[validation_positions])
        train_subject_ids = tuple(
            sorted(map(int, np.unique(eligible_groups[train_positions])))
        )
        validation_subject_ids = tuple(
            sorted(map(int, np.unique(eligible_groups[validation_positions])))
        )
        folds.append(
            InnerCvFold(
                train_graph_indices=train_graph_indices,
                validation_graph_indices=validation_graph_indices,
                train_subject_ids=train_subject_ids,
                validation_subject_ids=validation_subject_ids,
                fold=fold,
            )
        )

    result = tuple(folds)
    _validate_inner_folds(result, values, eligible)
    return result
