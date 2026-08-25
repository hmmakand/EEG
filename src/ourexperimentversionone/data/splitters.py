"""Leakage-safe split generation for the Liu2024 EEG dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass(frozen=True)
class DataSplit:
    """Window indices and original subject IDs for one evaluation fold."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_subject_ids: tuple[int, ...]
    validation_subject_ids: tuple[int, ...]
    test_subject_ids: tuple[int, ...]
    protocol: str
    fold: int


def _indices_for_subjects(subject_ids: np.ndarray, selected: tuple[int, ...]) -> np.ndarray:
    return np.flatnonzero(np.isin(subject_ids, selected))


def _unique_subjects(subject_ids: np.ndarray) -> np.ndarray:
    subjects = np.unique(subject_ids.astype(np.int64, copy=False))
    if len(subjects) < 2:
        raise ValueError("At least two subjects are required")
    return subjects


def _validate_cross_subject_split(split: DataSplit, all_indices: np.ndarray) -> None:
    index_sets = [
        set(map(int, split.train_indices)),
        set(map(int, split.validation_indices)),
        set(map(int, split.test_indices)),
    ]
    if any(index_sets[i] & index_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Split contains overlapping window indices")
    if set.union(*index_sets) != set(map(int, all_indices)):
        raise RuntimeError("Split does not cover every dataset window exactly once")

    subject_sets = [
        set(split.train_subject_ids),
        set(split.validation_subject_ids),
        set(split.test_subject_ids),
    ]
    if any(subject_sets[i] & subject_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Cross-subject split leaks subjects between partitions")


def group_kfold_splits(
    subject_ids: np.ndarray,
    *,
    n_splits: int = 5,
    validation_subjects: int = 8,
    seed: int = 42,
) -> tuple[DataSplit, ...]:
    """Create deterministic grouped folds with held-out validation subjects."""

    subjects = _unique_subjects(subject_ids)
    if not 2 <= n_splits <= len(subjects):
        raise ValueError("n_splits must be between 2 and the number of subjects")
    shuffled = np.random.default_rng(seed).permutation(subjects)
    test_groups = np.array_split(shuffled, n_splits)
    all_indices = np.arange(len(subject_ids))
    splits = []
    for fold, test_array in enumerate(test_groups):
        remaining = np.asarray([s for s in shuffled if s not in set(test_array)])
        if not 1 <= validation_subjects < len(remaining):
            raise ValueError("validation_subjects must leave at least one training subject")
        validation_array = np.random.default_rng(seed + fold + 1).permutation(remaining)[
            :validation_subjects
        ]
        validation_set = set(map(int, validation_array))
        train_array = np.asarray([s for s in remaining if int(s) not in validation_set])
        split = DataSplit(
            train_indices=_indices_for_subjects(subject_ids, tuple(map(int, train_array))),
            validation_indices=_indices_for_subjects(
                subject_ids, tuple(map(int, validation_array))
            ),
            test_indices=_indices_for_subjects(subject_ids, tuple(map(int, test_array))),
            train_subject_ids=tuple(sorted(map(int, train_array))),
            validation_subject_ids=tuple(sorted(map(int, validation_array))),
            test_subject_ids=tuple(sorted(map(int, test_array))),
            protocol="group_kfold",
            fold=fold,
        )
        _validate_cross_subject_split(split, all_indices)
        splits.append(split)
    return tuple(splits)


def loso_split(
    subject_ids: np.ndarray,
    *,
    test_subject_id: int,
) -> DataSplit:
    """Hold out one test subject and train on every other subject."""

    subjects = _unique_subjects(subject_ids)
    subject_list = list(map(int, subjects))
    if test_subject_id not in subject_list:
        raise ValueError(f"Unknown test subject: {test_subject_id}")
    test_position = subject_list.index(test_subject_id)
    train = tuple(subject for subject in subject_list if subject != test_subject_id)
    split = DataSplit(
        train_indices=_indices_for_subjects(subject_ids, train),
        validation_indices=np.asarray([], dtype=np.int64),
        test_indices=_indices_for_subjects(subject_ids, (test_subject_id,)),
        train_subject_ids=train,
        validation_subject_ids=(),
        test_subject_ids=(test_subject_id,),
        protocol="loso",
        fold=test_position,
    )
    train_set = set(map(int, split.train_indices))
    test_set = set(map(int, split.test_indices))
    if train_set & test_set:
        raise RuntimeError("LOSO split contains overlapping window indices")
    if train_set | test_set != set(range(len(subject_ids))):
        raise RuntimeError("LOSO split does not cover every dataset window")
    if set(split.train_subject_ids) & set(split.test_subject_ids):
        raise RuntimeError("LOSO split leaks the test subject into training")
    return split


def within_subject_splits(
    subject_ids: np.ndarray,
    targets: np.ndarray,
    *,
    subject_id: int,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[DataSplit, ...]:
    """Create stratified train/validation/test trial folds for one subject."""

    subject_indices = np.flatnonzero(subject_ids == subject_id)
    if len(subject_indices) == 0:
        raise ValueError(f"Unknown subject: {subject_id}")
    subject_targets = np.asarray(targets)[subject_indices]
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold, (development_local, test_local) in enumerate(
        outer.split(subject_indices, subject_targets)
    ):
        development_indices = subject_indices[development_local]
        development_targets = np.asarray(targets)[development_indices]
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=0.25, random_state=seed + fold + 1
        )
        train_local, validation_local = next(
            inner.split(development_indices, development_targets)
        )
        train_indices = development_indices[train_local]
        validation_indices = development_indices[validation_local]
        test_indices = subject_indices[test_local]
        if set(map(int, train_indices)) & set(map(int, validation_indices)):
            raise RuntimeError("Within-subject split contains overlapping indices")
        splits.append(
            DataSplit(
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
                train_subject_ids=(subject_id,),
                validation_subject_ids=(subject_id,),
                test_subject_ids=(subject_id,),
                protocol="within_subject",
                fold=fold,
            )
        )
    return tuple(splits)
