"""Dataset splitting utilities built around Braindecode protocol splits.

For BCI Competition IV 2a / MOABB BNCI2014_001, Braindecode exposes the
recording protocol in the dataset description. The important convention is:

- session=0train is the training pool.
- session=1test is the final test set.

All validation, cross-validation, and grid-search resampling must happen inside
train_pool. The final test_set is never used for model selection.

The random split helpers are kept for synthetic smoke-test datasets, but real
Braindecode protocol datasets should use make_braindecode_protocol_split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import torch
from braindecode.datasets import BaseConcatDataset
from omegaconf import DictConfig
from sklearn.model_selection import BaseCrossValidator, KFold
from torch.utils.data import Dataset, Subset, random_split

from eeg_bci.data.adapters import BraindecodeLikeDataset, TensorDatasetFromBraindecode

SESSION_TRAIN_TEST = "session_train_test"
SESSION_TRAIN_VALID_TEST = "session_train_valid_test"
SESSION_CROSS_VALIDATION_TEST = "session_cross_validation_test"
SESSION_GRID_SEARCH_TEST = "session_grid_search_test"
LEAVE_ONE_SUBJECT_OUT = "leave_one_subject_out"

SESSION_SPLIT_STRATEGIES = {
    SESSION_TRAIN_TEST,
    SESSION_TRAIN_VALID_TEST,
    SESSION_CROSS_VALIDATION_TEST,
    SESSION_GRID_SEARCH_TEST,
}

Resampler: TypeAlias = KFold | BaseCrossValidator


@dataclass(frozen=True)
class SplitPlan:
    """Resolved datasets and optional resampler for one split strategy."""

    split_strategy: str
    train_pool: Dataset
    train_set: Dataset
    valid_set: Dataset | None
    test_set: Dataset
    resampler: Resampler | None


@dataclass(frozen=True)
class CrossSubjectFold:
    """One leave-one-subject-out fold."""

    held_out_subject: int | str
    train_subjects: list[int | str]
    train_set: Dataset
    valid_set: Dataset | None
    test_set: Dataset


class SplittableBraindecodeDataset(Protocol):
    """Protocol for Braindecode datasets that support metadata splitting."""

    def split(self, by: str | None = None, **kwargs: Any) -> dict[str, Any]: ...


class SizedDataset(Protocol):
    def __len__(self) -> int: ...


class HoldoutSplit(BaseCrossValidator):
    """One deterministic train/validation split for sklearn CV APIs."""

    def __init__(self, *, valid_size: float, shuffle: bool, seed: int) -> None:
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.seed = seed

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return 1

    def split(self, X: Any, y: Any = None, groups: Any = None):
        train_len, _ = _split_lengths(len(X), holdout_size=self.valid_size)
        indices = np.arange(len(X))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        yield indices[:train_len], indices[train_len:]


def make_braindecode_protocol_split(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
    *,
    seed: int,
) -> SplitPlan:
    """Create a split plan from a Braindecode description/session protocol."""

    split_strategy = _session_split_strategy(split_cfg)
    train_pool, test_set = split_by_description(dataset, split_cfg)

    if split_strategy == SESSION_TRAIN_TEST:
        return SplitPlan(
            split_strategy=split_strategy,
            train_pool=train_pool,
            train_set=train_pool,
            valid_set=None,
            test_set=test_set,
            resampler=None,
        )

    if split_strategy == SESSION_TRAIN_VALID_TEST:
        train_set, valid_set = split_train_valid(
            train_pool,
            valid_size=_validation_size(split_cfg),
            shuffle=_validation_shuffle(split_cfg),
            seed=seed,
        )
        return SplitPlan(
            split_strategy=split_strategy,
            train_pool=train_pool,
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            resampler=None,
        )

    if split_strategy in {SESSION_CROSS_VALIDATION_TEST, SESSION_GRID_SEARCH_TEST}:
        resampler = make_resampler(split_cfg, seed=seed)
        return SplitPlan(
            split_strategy=split_strategy,
            train_pool=train_pool,
            train_set=train_pool,
            valid_set=None,
            test_set=test_set,
            resampler=resampler,
        )

    available = ", ".join(sorted(SESSION_SPLIT_STRATEGIES))
    raise ValueError(
        f"Unsupported session split strategy {split_strategy}. Available: {available}."
    )


def split_train_test(
    dataset: BraindecodeLikeDataset,
    dataset_cfg: DictConfig,
    *,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into training and final test sets."""

    split_cfg = dataset_cfg.get("split", None)
    strategy = _configured_strategy(split_cfg)

    if strategy in SESSION_SPLIT_STRATEGIES or strategy == "description":
        split_plan = make_braindecode_protocol_split(dataset, split_cfg, seed=seed)
        return split_plan.train_pool, split_plan.test_set

    if strategy == "random":
        test_size = (
            float(split_cfg.get("test_size", dataset_cfg.get("test_size", 0.2)))
            if split_cfg is not None
            else float(dataset_cfg.get("test_size", 0.2))
        )
        return split_random_train_test(dataset, test_size=test_size, seed=seed)

    raise ValueError(f"Unsupported split strategy {strategy}.")


def split_train_eval(
    dataset: BraindecodeLikeDataset,
    dataset_cfg: DictConfig,
    *,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Backward-compatible alias for split_train_test."""

    return split_train_test(dataset, dataset_cfg, seed=seed)


def split_random_train_test(
    dataset: BraindecodeLikeDataset,
    *,
    test_size: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Create a deterministic random train/test split for smoke-test datasets."""

    tensor_dataset = TensorDatasetFromBraindecode(dataset)
    train_len, test_len = _split_lengths(len(tensor_dataset), holdout_size=test_size)
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(
        tensor_dataset,
        [train_len, test_len],
        generator=generator,
    )
    return train_set, test_set


def split_train_valid(
    train_set: Dataset,
    *,
    valid_size: float,
    seed: int,
    shuffle: bool,
) -> tuple[Dataset, Dataset]:
    """Split a training set into inner-training and validation subsets."""

    sized_train_set = cast(SizedDataset, train_set)
    train_len, valid_len = _split_lengths(len(sized_train_set), holdout_size=valid_size)

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


def split_by_description(
    dataset: BraindecodeLikeDataset, split_cfg: DictConfig
) -> tuple[Dataset, Dataset]:
    """Split a Braindecode dataset using description metadata."""

    train_source, test_source = _split_by_session_description(dataset, split_cfg)
    return (
        TensorDatasetFromBraindecode(train_source),
        TensorDatasetFromBraindecode(test_source),
    )


def make_leave_one_subject_out_folds(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
    *,
    seed: int,
) -> list[CrossSubjectFold]:
    """Create train/test-only leave-one-subject-out folds."""

    _ = seed
    train_pool, test_pool = _split_by_session_description(dataset, split_cfg)
    subject_cfg = _section(split_cfg, "subject")
    subject_column = str(
        subject_cfg.get(
            "column",
            split_cfg.get("subject_column", "subject"),
        )
    )

    train_by_subject = _description_splits(train_pool, subject_column)
    test_by_subject = _description_splits(test_pool, subject_column)
    subject_keys = sorted(
        set(train_by_subject).intersection(test_by_subject),
        key=_sort_description_key,
    )
    if len(subject_keys) < 2:
        raise ValueError(
            "Leave-one-subject-out requires at least two subjects present in both "
            f"the training and test sessions for column {subject_column}."
        )

    folds: list[CrossSubjectFold] = []
    for held_out_key in subject_keys:
        train_keys = [key for key in subject_keys if key != held_out_key]
        train_source = _concat_braindecode_sources(
            [train_by_subject[key] for key in train_keys]
        )
        test_source = test_by_subject[held_out_key]
        folds.append(
            CrossSubjectFold(
                held_out_subject=_normalize_description_key(held_out_key),
                train_subjects=[_normalize_description_key(key) for key in train_keys],
                train_set=TensorDatasetFromBraindecode(train_source),
                valid_set=None,
                test_set=TensorDatasetFromBraindecode(test_source),
            )
        )
    return folds


def _split_by_session_description(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
) -> tuple[BraindecodeLikeDataset, BraindecodeLikeDataset]:
    session_cfg = _section(split_cfg, "session")
    column = str(session_cfg.get("column", split_cfg.get("column", "session")))
    train_key = str(session_cfg.get("train_key", split_cfg.get("train_key", "0train")))
    test_key = str(
        session_cfg.get(
            "test_key",
            split_cfg.get("test_key", split_cfg.get("eval_key", "1test")),
        )
    )

    splits = _description_splits(dataset, column)
    missing_keys = [key for key in (train_key, test_key) if key not in splits]
    if missing_keys:
        available = ", ".join(str(key) for key in sorted(splits, key=_sort_description_key))
        missing = ", ".join(missing_keys)
        raise KeyError(
            f"Split keys not found for column {column}: {missing}. "
            f"Available keys: {available}."
        )

    return splits[train_key], splits[test_key]


def _description_splits(
    dataset: BraindecodeLikeDataset,
    column: str,
) -> dict[Any, BraindecodeLikeDataset]:
    if not hasattr(dataset, "split"):
        raise TypeError(
            "Description-based splitting requires a Braindecode dataset with split()."
        )
    splittable = cast(SplittableBraindecodeDataset, dataset)
    return splittable.split(column)


def _concat_braindecode_sources(
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


def _normalize_description_key(key: Any) -> int | str:
    if hasattr(key, "item"):
        key = key.item()
    try:
        return int(key)
    except (TypeError, ValueError):
        return str(key)


def _sort_description_key(key: Any) -> tuple[int, int | str]:
    normalized = _normalize_description_key(key)
    if isinstance(normalized, int):
        return (0, normalized)
    return (1, normalized)


def make_resampler(split_cfg: DictConfig, *, seed: int) -> Resampler:
    """Create the resampler used inside the training pool."""

    resampling_cfg = _section(split_cfg, "resampling")
    method = str(resampling_cfg.get("method", "kfold"))
    shuffle = bool(resampling_cfg.get("shuffle", split_cfg.get("shuffle", False)))

    if method == "kfold":
        n_splits = int(resampling_cfg.get("n_splits", split_cfg.get("n_splits", 5)))
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        random_state = seed if shuffle else None
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    if method == "holdout":
        valid_size = float(resampling_cfg.get("valid_size", _validation_size(split_cfg)))
        return HoldoutSplit(valid_size=valid_size, shuffle=shuffle, seed=seed)

    raise ValueError(f"Unsupported resampling method {method}.")


def _validation_size(split_cfg: DictConfig) -> float:
    validation_cfg = _section(split_cfg, "validation")
    return float(
        validation_cfg.get(
            "valid_size",
            split_cfg.get("valid_size", split_cfg.get("validation_size", 0.2)),
        )
    )


def _validation_shuffle(split_cfg: DictConfig) -> bool:
    validation_cfg = _section(split_cfg, "validation")
    return bool(
        validation_cfg.get(
            "shuffle",
            split_cfg.get("shuffle", split_cfg.get("validation_shuffle", False)),
        )
    )


def _section(split_cfg: DictConfig, name: str) -> DictConfig:
    value = split_cfg.get(name, None)
    return value if isinstance(value, DictConfig) else split_cfg


def _configured_strategy(split_cfg: DictConfig | None) -> str:
    if split_cfg is None:
        return "random"
    return str(split_cfg.get("split_strategy", split_cfg.get("strategy", "random")))


def _session_split_strategy(split_cfg: DictConfig) -> str:
    strategy = _configured_strategy(split_cfg)
    if strategy == "description":
        return SESSION_TRAIN_TEST
    return strategy


def _split_lengths(total_len: int, *, holdout_size: float) -> tuple[int, int]:
    if total_len < 2:
        raise ValueError("At least two samples are required to create a split.")
    if not 0.0 < holdout_size < 1.0:
        raise ValueError("Holdout size must be between 0 and 1.")

    train_len = int(round(total_len * (1.0 - holdout_size)))
    train_len = min(max(train_len, 1), total_len - 1)
    holdout_len = total_len - train_len
    return train_len, holdout_len
