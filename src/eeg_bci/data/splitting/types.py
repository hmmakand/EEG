"""Shared split data structures and protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold
from torch.utils.data import Dataset

Resampler: TypeAlias = KFold | BaseCrossValidator


@dataclass(frozen=True)
class SplitSource:
    """Outer train/test datasets before choosing the training methodology."""

    train_pool: Dataset
    test_set: Dataset
    groups: np.ndarray | None = None
    """Per-row subject id for ``train_pool``, in the same row order. ``None``
    for sources with no subject concept (e.g. synthetic/random). Used to keep
    validation splits and resampling folds proportionally representative of
    every pooled subject instead of taking a positional slice that can land
    entirely within one subject."""


@dataclass(frozen=True)
class SplitPlan:
    """Resolved datasets and optional resampler for one split strategy."""

    split_strategy: str
    method: str
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
    split_plan: SplitPlan

    @property
    def train_set(self) -> Dataset:
        return self.split_plan.train_set

    @property
    def valid_set(self) -> Dataset | None:
        return self.split_plan.valid_set

    @property
    def test_set(self) -> Dataset:
        return self.split_plan.test_set


class SplittableBraindecodeDataset(Protocol):
    """Protocol for Braindecode datasets that support metadata splitting."""

    def split(self, by: str | None = None, **kwargs: Any) -> dict[str, Any]: ...


class SizedDataset(Protocol):
    def __len__(self) -> int: ...
