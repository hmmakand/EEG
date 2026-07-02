from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from eeg_bci.data.splitting.config import grouped_split_indices
from eeg_bci.data.splitting.plans import split_train_valid
from eeg_bci.data.splitting.resampling import HoldoutSplit, PerGroupKFold
from eeg_bci.data.splitting.sources import make_chronological_split_source


def _dataset(n: int) -> TensorDataset:
    return TensorDataset(torch.arange(n))


def test_grouped_split_indices_gives_every_group_proportional_share() -> None:
    groups = np.array([1] * 10 + [2] * 10)

    train_idx, valid_idx = grouped_split_indices(
        20, groups, holdout_size=0.2, shuffle=False, seed=0
    )

    assert len(valid_idx) == 4
    assert sorted(set(groups[valid_idx])) == [1, 2]
    for group in (1, 2):
        assert (groups[valid_idx] == group).sum() == 2
        assert (groups[train_idx] == group).sum() == 8


def test_split_train_valid_without_groups_is_unchanged() -> None:
    dataset = _dataset(10)

    train_set, valid_set = split_train_valid(dataset, valid_size=0.2, seed=0, shuffle=False)

    assert len(train_set) == 8
    assert len(valid_set) == 2
    assert list(valid_set.indices) == [8, 9]


def test_split_train_valid_with_groups_avoids_single_subject_validation_set() -> None:
    # Rows ordered subject-by-subject, as pooled multi-subject data is.
    groups = np.array([1] * 10 + [2] * 10 + [3] * 10)
    dataset = _dataset(30)

    train_set, valid_set = split_train_valid(
        dataset, valid_size=0.2, seed=0, shuffle=False, groups=groups
    )

    valid_groups = groups[valid_set.indices]
    train_groups = groups[train_set.indices]
    assert sorted(set(valid_groups)) == [1, 2, 3]
    for subject in (1, 2, 3):
        assert (valid_groups == subject).sum() == 2
        assert (train_groups == subject).sum() == 8


def test_per_group_kfold_distributes_every_group_across_every_fold() -> None:
    groups = np.array([1] * 10 + [2] * 10 + [3] * 10)
    resampler = PerGroupKFold(n_splits=5, shuffle=False, seed=0, groups=groups)

    folds = list(resampler.split(np.zeros((30, 1))))
    assert len(folds) == 5
    for train_idx, valid_idx in folds:
        valid_groups = groups[valid_idx]
        assert sorted(set(valid_groups)) == [1, 2, 3]
        for subject in (1, 2, 3):
            assert (valid_groups == subject).sum() == 2
        assert sorted(set(groups[train_idx])) == [1, 2, 3]


def test_per_group_kfold_rejects_groups_smaller_than_n_splits() -> None:
    groups = np.array([1] * 10 + [2] * 3)

    with pytest.raises(ValueError, match="fewer"):
        PerGroupKFold(n_splits=5, shuffle=False, seed=0, groups=groups)


def test_holdout_split_with_groups_avoids_single_subject_validation_set() -> None:
    groups = np.array([1] * 10 + [2] * 10)
    resampler = HoldoutSplit(valid_size=0.2, shuffle=False, seed=0, groups=groups)

    (train_idx, valid_idx) = next(resampler.split(np.zeros((20, 1))))
    valid_groups = groups[valid_idx]
    assert sorted(set(valid_groups)) == [1, 2]
    for subject in (1, 2):
        assert (valid_groups == subject).sum() == 2


class _FakeMetadataDataset:
    def __init__(self, metadata: pd.DataFrame) -> None:
        self._metadata = metadata

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, None]:
        return torch.zeros(1), int(self._metadata.iloc[index]["target"]), None

    def get_metadata(self) -> pd.DataFrame:
        return self._metadata


def _trials(subject: int, n_per_class: int, *, start: int, step: int = 1000) -> pd.DataFrame:
    rows = []
    offset = start
    for label in (0, 1):
        for _ in range(n_per_class):
            rows.append({"subject": subject, "target": label, "i_start_in_trial": offset})
            offset += step
    return pd.DataFrame(rows)


def test_chronological_split_source_populates_groups_for_pooled_subjects() -> None:
    metadata = pd.concat(
        [
            _trials(subject=1, n_per_class=10, start=0),
            _trials(subject=2, n_per_class=10, start=1_000_000),
        ],
        ignore_index=True,
    )
    dataset = _FakeMetadataDataset(metadata)
    split_cfg = OmegaConf.create({"test_size": 0.2, "stratify": True})

    source = make_chronological_split_source(dataset, split_cfg)

    assert source.groups is not None
    assert len(source.groups) == len(source.train_pool)
    assert sorted(set(source.groups.tolist())) == [1, 2]
