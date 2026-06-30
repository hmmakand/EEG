"""Leave-one-subject-out split construction."""

from __future__ import annotations

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset

from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.splitting.config import resolved_source_and_method, section
from eeg_bci.data.splitting.description import (
    description_splits,
    normalize_description_key,
    sort_description_key,
)
from eeg_bci.data.splitting.plans import make_loso_plan, split_train_valid
from eeg_bci.data.splitting.sources import build_split_source
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
    """Create leave-one-subject-out folds for any registered split source."""

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

    subject_sources = {
        key: build_split_source(source, sub_dataset, split_cfg, seed=seed)
        for key, sub_dataset in by_subject.items()
    }

    folds: list[CrossSubjectFold] = []
    for held_out_key in subject_keys:
        train_keys = [key for key in subject_keys if key != held_out_key]
        train_set: Dataset = ConcatDataset(
            [subject_sources[key].train_pool for key in train_keys]
        )
        valid_set: Dataset | None = None
        if validation_enabled:
            train_set, valid_set = split_train_valid(
                train_set,
                valid_size=validation_size,
                shuffle=validation_shuffle,
                seed=seed,
            )
        test_set = subject_sources[held_out_key].test_set
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
