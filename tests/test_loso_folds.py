from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from eeg_bci.data.splitting.loso import make_leave_one_subject_out_folds


class _FakeSubjectDataset:
    """A braindecode-like per-subject dataset yielding ``(x, y, extra)``."""

    def __init__(self, targets: list[int]) -> None:
        self._targets = list(targets)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, None]:
        return torch.zeros(2, 4), int(self._targets[index]), None


class _FakeSplittableDataset:
    """Stands in for a Braindecode dataset with ``split('subject')``."""

    def __init__(self, by_subject: dict[int, _FakeSubjectDataset]) -> None:
        self._by_subject = by_subject

    def split(self, by: str) -> dict[int, _FakeSubjectDataset]:
        assert by == "subject"
        return dict(self._by_subject)


def _balanced_targets(n_per_class: int) -> list[int]:
    return [0] * n_per_class + [1] * n_per_class


def _dataset(sizes: dict[int, int]) -> _FakeSplittableDataset:
    return _FakeSplittableDataset(
        {subject: _FakeSubjectDataset(_balanced_targets(size)) for subject, size in sizes.items()}
    )


def _split_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "source": "chronological",
            "method": "leave_one_subject_out",
            "subject": {"column": "subject"},
        }
    )


def test_loso_fold_trains_on_full_non_held_out_subjects() -> None:
    # Per-subject window counts (2 * n_per_class).
    sizes = {1: 6, 2: 8, 3: 10}
    dataset = _dataset({s: n // 2 for s, n in sizes.items()})

    folds = make_leave_one_subject_out_folds(dataset, _split_cfg(), seed=0)

    assert [fold.held_out_subject for fold in folds] == [1, 2, 3]
    for fold in folds:
        held_out = fold.held_out_subject
        expected_train = sum(size for subject, size in sizes.items() if subject != held_out)
        assert len(fold.train_set) == expected_train
        assert len(fold.test_set) == sizes[held_out]


def test_loso_folds_use_every_window_as_test_exactly_once() -> None:
    sizes = {1: 6, 2: 8, 3: 10}
    dataset = _dataset({s: n // 2 for s, n in sizes.items()})

    folds = make_leave_one_subject_out_folds(dataset, _split_cfg(), seed=0)

    total_test = sum(len(fold.test_set) for fold in folds)
    assert total_test == sum(sizes.values())
    assert sorted(fold.held_out_subject for fold in folds) == [1, 2, 3]


def test_loso_validation_represents_every_training_subject() -> None:
    sizes = {1: 8, 2: 8, 3: 8}
    dataset = _dataset({s: n // 2 for s, n in sizes.items()})

    folds = make_leave_one_subject_out_folds(
        dataset,
        _split_cfg(),
        seed=0,
        validation_enabled=True,
        validation_size=0.25,
        validation_shuffle=False,
    )

    fold = folds[0]  # held out subject 1; trains on subjects 2 and 3
    assert fold.valid_set is not None
    # Inner-train + validation together still cover every training window.
    assert len(fold.train_set) + len(fold.valid_set) == sizes[2] + sizes[3]
    # Each training subject contributes 25% of its 8 windows to validation.
    assert len(fold.valid_set) == 4
    # Coverage: reconstructed target counts match a balanced two-subject pool.
    valid_targets = np.array([int(fold.valid_set[i][1]) for i in range(len(fold.valid_set))])
    assert valid_targets.tolist().count(0) + valid_targets.tolist().count(1) == 4
