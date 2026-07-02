"""Dataset splitting utilities built around outer splits and split plans.

A dataset's ``split`` config sets two independent, orthogonal keys:

- ``source`` -- how train_pool/test_set are carved out of the raw dataset
  (dataset-structure-dependent, e.g. ``session`` for BCI IV 2a's official
  ``0train``/``1test`` recording protocol, ``chronological`` for datasets
  without a session column).
- ``method`` -- the dataset-agnostic training/evaluation methodology applied
  on top of the resulting (train_pool, test_set) pair (``train_test``,
  ``train_valid_test``, ``cross_validation_test``, ``grid_search_test``,
  ``leave_one_subject_out``).

All validation, cross-validation, and grid-search resampling happens inside
train_pool. The final test_set is never used for model selection.
"""

from __future__ import annotations

from eeg_bci.data.splitting.config import resolved_source_and_method, resolved_split_label
from eeg_bci.data.splitting.loso import make_leave_one_subject_out_folds
from eeg_bci.data.splitting.plans import (
    make_grid_search_plan,
    make_loso_plan,
    make_protocol_split,
    make_resampled_plan,
    make_split_plan,
    make_train_test_plan,
    make_train_valid_test_plan,
    split_train_valid,
)
from eeg_bci.data.splitting.resampling import (
    HoldoutSplit,
    PerGroupKFold,
    make_chronological_resampler,
    make_resampler,
)
from eeg_bci.data.splitting.sources import (
    SOURCE_BUILDERS,
    build_split_source,
    split_by_description,
    split_chronological_train_test,
    split_random_train_test,
)
from eeg_bci.data.splitting.strategies import (
    CROSS_VALIDATION_TEST,
    GRID_SEARCH_TEST,
    LOSO,
    METHODS,
    SOURCE_CHRONOLOGICAL,
    SOURCE_RANDOM,
    SOURCE_SESSION,
    TRAIN_TEST,
    TRAIN_VALID_TEST,
    split_label,
)
from eeg_bci.data.splitting.types import (
    CrossSubjectFold,
    Resampler,
    SplitPlan,
    SplitSource,
)

__all__ = [
    "CROSS_VALIDATION_TEST",
    "CrossSubjectFold",
    "GRID_SEARCH_TEST",
    "HoldoutSplit",
    "LOSO",
    "METHODS",
    "PerGroupKFold",
    "Resampler",
    "SOURCE_BUILDERS",
    "SOURCE_CHRONOLOGICAL",
    "SOURCE_RANDOM",
    "SOURCE_SESSION",
    "SplitPlan",
    "SplitSource",
    "TRAIN_TEST",
    "TRAIN_VALID_TEST",
    "build_split_source",
    "make_chronological_resampler",
    "make_grid_search_plan",
    "make_leave_one_subject_out_folds",
    "make_loso_plan",
    "make_protocol_split",
    "make_resampled_plan",
    "make_resampler",
    "make_split_plan",
    "make_train_test_plan",
    "make_train_valid_test_plan",
    "resolved_source_and_method",
    "resolved_split_label",
    "split_by_description",
    "split_chronological_train_test",
    "split_label",
    "split_random_train_test",
    "split_train_valid",
]
