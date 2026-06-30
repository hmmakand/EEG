from __future__ import annotations

import pandas as pd
import torch
from omegaconf import OmegaConf

from eeg_bci.data.splitting.sources import make_chronological_split_source


class _FakeMetadataDataset:
    """Minimal Braindecode-like dataset backed by a metadata DataFrame."""

    def __init__(self, metadata: pd.DataFrame) -> None:
        self._metadata = metadata

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, None]:
        return torch.zeros(1), int(self._metadata.iloc[index]["target"]), None

    def get_metadata(self) -> pd.DataFrame:
        return self._metadata


def _trials(subject: int, n_per_class: int, *, start: int, step: int = 1000) -> pd.DataFrame:
    """Build per-class trials for one subject with increasing i_start_in_trial."""

    rows = []
    offset = start
    for label in (0, 1):
        for _ in range(n_per_class):
            rows.append({"subject": subject, "target": label, "i_start_in_trial": offset})
            offset += step
    return pd.DataFrame(rows)


def _split(metadata: pd.DataFrame, *, test_size: float = 0.2, stratify: bool = True):
    dataset = _FakeMetadataDataset(metadata)
    split_cfg = OmegaConf.create({"test_size": test_size, "stratify": stratify})
    source = make_chronological_split_source(dataset, split_cfg)
    return metadata.iloc[source.train_pool.indices], metadata.iloc[source.test_set.indices]


def test_single_subject_takes_latest_trials_per_class_as_test() -> None:
    metadata = _trials(subject=1, n_per_class=10, start=0)

    train, test = _split(metadata)

    assert len(test) == 4  # 2 per class
    assert len(train) == 16
    for label in (0, 1):
        class_train = train[train["target"] == label]
        class_test = test[test["target"] == label]
        assert class_train["i_start_in_trial"].max() < class_test["i_start_in_trial"].min()


def test_multiple_subjects_with_matching_ranges_each_get_proportional_test_trials() -> None:
    metadata = pd.concat(
        [
            _trials(subject=1, n_per_class=10, start=1000),
            _trials(subject=2, n_per_class=10, start=1000),
        ],
        ignore_index=True,
    )

    train, test = _split(metadata)

    for subject in (1, 2):
        assert len(test[test["subject"] == subject]) == 4
        assert len(train[train["subject"] == subject]) == 16


def test_multiple_subjects_with_disjoint_ranges_each_still_get_their_own_test_trials() -> None:
    # Subject 2's recording-local offsets are entirely larger than subject 1's,
    # even though the two recordings are unrelated in real time. A global sort
    # across subjects would put subject 1 entirely in train and subject 2
    # entirely in test; the per-subject split must avoid that.
    metadata = pd.concat(
        [
            _trials(subject=1, n_per_class=10, start=0),
            _trials(subject=2, n_per_class=10, start=1_000_000),
        ],
        ignore_index=True,
    )

    train, test = _split(metadata)

    for subject in (1, 2):
        subject_test = test[test["subject"] == subject]
        subject_train = train[train["subject"] == subject]
        assert len(subject_test) == 4
        assert len(subject_train) == 16
        for label in (0, 1):
            assert (
                subject_train[subject_train["target"] == label]["i_start_in_trial"].max()
                < subject_test[subject_test["target"] == label]["i_start_in_trial"].min()
            )


def test_non_stratified_split_is_also_computed_per_subject() -> None:
    metadata = pd.concat(
        [
            _trials(subject=1, n_per_class=10, start=0),
            _trials(subject=2, n_per_class=10, start=1_000_000),
        ],
        ignore_index=True,
    )

    train, test = _split(metadata, stratify=False)

    for subject in (1, 2):
        assert len(test[test["subject"] == subject]) > 0
        assert len(train[train["subject"] == subject]) > 0
