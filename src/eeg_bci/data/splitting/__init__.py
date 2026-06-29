"""Dataset splitting utilities built around outer splits and split plans.

For BCI Competition IV 2a / MOABB BNCI2014_001, Braindecode exposes the
recording protocol in the dataset description. The important convention is:

- session=0train is the training pool.
- session=1test is the final test set.

All validation, cross-validation, and grid-search resampling must happen inside
train_pool. The final test_set is never used for model selection.
"""

from __future__ import annotations

from eeg_bci.data.splitting.api import split_train_eval, split_train_test
from eeg_bci.data.splitting.loso import make_leave_one_subject_out_folds
from eeg_bci.data.splitting.plans import (
    make_braindecode_protocol_split,
    make_chronological_protocol_split,
    make_grid_search_plan,
    make_loso_plan,
    make_resampled_plan,
    make_split_plan,
    make_train_test_plan,
    make_train_valid_test_plan,
    split_train_valid,
)
from eeg_bci.data.splitting.resampling import (
    HoldoutSplit,
    make_chronological_resampler,
    make_resampler,
)
from eeg_bci.data.splitting.sources import (
    split_by_description,
    split_chronological_train_test,
    split_random_train_test,
)
from eeg_bci.data.splitting.strategies import (
    CHRONOLOGICAL_CROSS_VALIDATION_TEST,
    CHRONOLOGICAL_GRID_SEARCH_TEST,
    CHRONOLOGICAL_LEAVE_ONE_SUBJECT_OUT,
    CHRONOLOGICAL_SPLIT_STRATEGIES,
    CHRONOLOGICAL_TRAIN_TEST,
    CHRONOLOGICAL_TRAIN_VALID_TEST,
    CROSS_VALIDATION_TEST,
    GRID_SEARCH_TEST,
    LOSO_SPLIT_STRATEGIES,
    LOSO,
    SESSION_CROSS_VALIDATION_TEST,
    SESSION_GRID_SEARCH_TEST,
    SESSION_LEAVE_ONE_SUBJECT_OUT,
    SESSION_SPLIT_STRATEGIES,
    SESSION_TRAIN_TEST,
    SESSION_TRAIN_VALID_TEST,
    TRAIN_TEST,
    TRAIN_VALID_TEST,
    split_method,
)
from eeg_bci.data.splitting.types import (
    CrossSubjectFold,
    Resampler,
    SplitPlan,
    SplitSource,
)

__all__ = [
    "CHRONOLOGICAL_CROSS_VALIDATION_TEST",
    "CHRONOLOGICAL_GRID_SEARCH_TEST",
    "CHRONOLOGICAL_LEAVE_ONE_SUBJECT_OUT",
    "CHRONOLOGICAL_SPLIT_STRATEGIES",
    "CHRONOLOGICAL_TRAIN_TEST",
    "CHRONOLOGICAL_TRAIN_VALID_TEST",
    "CROSS_VALIDATION_TEST",
    "CrossSubjectFold",
    "GRID_SEARCH_TEST",
    "LOSO_SPLIT_STRATEGIES",
    "HoldoutSplit",
    "LOSO",
    "Resampler",
    "SESSION_CROSS_VALIDATION_TEST",
    "SESSION_GRID_SEARCH_TEST",
    "SESSION_LEAVE_ONE_SUBJECT_OUT",
    "SESSION_SPLIT_STRATEGIES",
    "SESSION_TRAIN_TEST",
    "SESSION_TRAIN_VALID_TEST",
    "SplitPlan",
    "SplitSource",
    "TRAIN_TEST",
    "TRAIN_VALID_TEST",
    "make_braindecode_protocol_split",
    "make_chronological_protocol_split",
    "make_chronological_resampler",
    "make_grid_search_plan",
    "make_leave_one_subject_out_folds",
    "make_loso_plan",
    "make_resampled_plan",
    "make_resampler",
    "make_split_plan",
    "make_train_test_plan",
    "make_train_valid_test_plan",
    "split_by_description",
    "split_chronological_train_test",
    "split_method",
    "split_random_train_test",
    "split_train_eval",
    "split_train_test",
    "split_train_valid",
]
