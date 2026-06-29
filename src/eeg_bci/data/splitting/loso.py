"""Leave-one-subject-out split construction."""

from __future__ import annotations

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset

from eeg_bci.data.adapters import BraindecodeLikeDataset, TensorDatasetFromBraindecode
from eeg_bci.data.splitting.config import configured_strategy, section
from eeg_bci.data.splitting.description import (
    concat_braindecode_sources,
    description_splits,
    normalize_description_key,
    sort_description_key,
    split_by_session_description,
)
from eeg_bci.data.splitting.plans import make_loso_plan, split_train_valid
from eeg_bci.data.splitting.sources import make_chronological_split_source
from eeg_bci.data.splitting.strategies import (
    CHRONOLOGICAL_LEAVE_ONE_SUBJECT_OUT,
    SESSION_LEAVE_ONE_SUBJECT_OUT,
)
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
    """Create leave-one-subject-out folds for the configured source split."""

    strategy = configured_strategy(split_cfg)
    if strategy == SESSION_LEAVE_ONE_SUBJECT_OUT:
        return make_session_leave_one_subject_out_folds(
            dataset,
            split_cfg,
            seed=seed,
            validation_enabled=validation_enabled,
            validation_size=validation_size,
            validation_shuffle=validation_shuffle,
        )
    if strategy == CHRONOLOGICAL_LEAVE_ONE_SUBJECT_OUT:
        return make_chronological_leave_one_subject_out_folds(
            dataset,
            split_cfg,
            seed=seed,
            validation_enabled=validation_enabled,
            validation_size=validation_size,
            validation_shuffle=validation_shuffle,
        )

    raise ValueError(
        "LOSO requires split.strategy to be one of "
        f"{SESSION_LEAVE_ONE_SUBJECT_OUT}, {CHRONOLOGICAL_LEAVE_ONE_SUBJECT_OUT}; "
        f"got {strategy}."
    )


def make_session_leave_one_subject_out_folds(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    validation_enabled: bool = False,
    validation_size: float = 0.2,
    validation_shuffle: bool = False,
) -> list[CrossSubjectFold]:
    """Create session-based leave-one-subject-out train/test folds."""

    train_pool, test_pool = split_by_session_description(dataset, split_cfg)
    subject_cfg = section(split_cfg, "subject")
    subject_column = str(
        subject_cfg.get(
            "column",
            split_cfg.get("subject_column", "subject"),
        )
    )

    train_by_subject = description_splits(train_pool, subject_column)
    test_by_subject = description_splits(test_pool, subject_column)
    subject_keys = sorted(
        set(train_by_subject).intersection(test_by_subject),
        key=sort_description_key,
    )
    if len(subject_keys) < 2:
        raise ValueError(
            "Leave-one-subject-out requires at least two subjects present in both "
            f"the training and test sessions for column {subject_column}."
        )

    folds: list[CrossSubjectFold] = []
    for held_out_key in subject_keys:
        train_keys = [key for key in subject_keys if key != held_out_key]
        train_source = concat_braindecode_sources(
            [train_by_subject[key] for key in train_keys]
        )
        test_source = test_by_subject[held_out_key]
        train_set = TensorDatasetFromBraindecode(train_source)
        valid_set: Dataset | None = None
        if validation_enabled:
            train_set, valid_set = split_train_valid(
                train_set,
                valid_size=validation_size,
                shuffle=validation_shuffle,
                seed=seed,
            )
        test_set = TensorDatasetFromBraindecode(test_source)
        split_plan = make_loso_plan(
            train_set,
            valid_set,
            test_set,
            strategy=SESSION_LEAVE_ONE_SUBJECT_OUT,
        )
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


def make_chronological_leave_one_subject_out_folds(
    dataset: BraindecodeLikeDataset,
    split_cfg: DictConfig,
    *,
    seed: int,
    validation_enabled: bool = False,
    validation_size: float = 0.2,
    validation_shuffle: bool = False,
) -> list[CrossSubjectFold]:
    """Create chronological leave-one-subject-out train/test folds."""

    subject_cfg = section(split_cfg, "subject")
    subject_column = str(
        subject_cfg.get(
            "column",
            split_cfg.get("subject_column", "subject"),
        )
    )
    by_subject = description_splits(dataset, subject_column)
    subject_keys = sorted(by_subject, key=sort_description_key)
    if len(subject_keys) < 2:
        raise ValueError(
            "Chronological leave-one-subject-out requires at least two subjects "
            f"for column {subject_column}."
        )

    subject_sources = {
        key: make_chronological_split_source(source, split_cfg)
        for key, source in by_subject.items()
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
        split_plan = make_loso_plan(
            train_set,
            valid_set,
            test_set,
            strategy=CHRONOLOGICAL_LEAVE_ONE_SUBJECT_OUT,
        )
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
