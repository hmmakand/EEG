"""Leave-one-subject-out split construction."""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset

from eeg_bci.data.adapters import BraindecodeLikeDataset, TensorDatasetFromBraindecode
from eeg_bci.data.splitting.config import resolved_source_and_method, section
from eeg_bci.data.splitting.description import (
    description_splits,
    normalize_description_key,
    sort_description_key,
)
from eeg_bci.data.splitting.plans import make_loso_plan, split_train_valid
from eeg_bci.data.splitting.strategies import LOSO
from eeg_bci.data.splitting.types import CrossSubjectFold


def make_leave_one_subject_out_folds(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    validation_enabled: bool = False,
    validation_size: float = 0.2,
    validation_shuffle: bool = False,
) -> list[CrossSubjectFold]:
    """Create leave-one-subject-out folds.

    Each fold trains on *all* windows of the non-held-out subjects and
    evaluates on *all* windows of the held-out subject (canonical LOSO). The
    intra-subject split ``source`` (e.g. ``session``/``chronological``) does
    not subset a subject inside a fold -- it is read only to validate the
    method and to label the run via ``split_label(source, LOSO)``.
    """

    source, method = resolved_source_and_method(split_cfg)
    if method != LOSO:
        raise ValueError(
            f"make_leave_one_subject_out_folds requires split.method={LOSO!r}; "
            f"got {method!r}."
        )

    subject_cfg = section(split_cfg, "subject")
    subject_column = str(
        subject_cfg.get("column", split_cfg.get("subject_column", "subject"))
    )

    by_subject = description_splits(dataset, subject_column)
    subject_keys = sorted(by_subject, key=sort_description_key)
    if len(subject_keys) < 2:
        raise ValueError(
            "Leave-one-subject-out requires at least two subjects present for "
            f"column {subject_column}."
        )

    subject_datasets: dict[Any, Dataset] = {
        key: TensorDatasetFromBraindecode(sub_dataset)
        for key, sub_dataset in by_subject.items()
    }

    folds: list[CrossSubjectFold] = []
    for held_out_key in subject_keys:
        train_keys = [key for key in subject_keys if key != held_out_key]
        train_set: Dataset = ConcatDataset(
            [subject_datasets[key] for key in train_keys]
        )
        valid_set: Dataset | None = None
        if validation_enabled:
            fold_groups = np.concatenate(
                [
                    np.full(
                        len(subject_datasets[key]),
                        normalize_description_key(key),
                    )
                    for key in train_keys
                ]
            )
            train_set, valid_set = split_train_valid(
                train_set,
                valid_size=validation_size,
                shuffle=validation_shuffle,
                seed=seed,
                groups=fold_groups,
            )
        test_set = subject_datasets[held_out_key]
        split_plan = make_loso_plan(train_set, valid_set, test_set, source=source)
        folds.append(
            CrossSubjectFold(
                held_out_subject=normalize_description_key(held_out_key),
                train_subjects=[
                    normalize_description_key(key) for key in train_keys
                ],
                split_plan=split_plan,
            )
        )
    return folds
