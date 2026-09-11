"""Within-subject split construction for the diagnostic classification tool.

Standalone module, not additions to :mod:`loso_split` -- see
``WITHIN_SUBJECT_PLAN.md`` for the full rationale. This operates on one
subject's own trials at a time: there is no train/validation/test *subject*
concept, no ``GraphDataLoaderConfig``, and no overlap in shape with
:mod:`loso_split`'s ``create_loso_splits``/``create_inner_cv_folds``, so it
does not belong alongside those types. ``loso_split.py`` stays untouched;
this module only imports its ``_graph_indices_for_subjects`` helper rather
than reimplementing "subject IDs -> graph indices" lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .loso_split import _graph_indices_for_subjects


@dataclass(frozen=True)
class WithinSubjectFold:
    """Graph indices assigned to one within-subject fold.

    Field name is deliberately ``evaluation_graph_indices``, not
    ``validation_...`` -- there is no early-stopping validation split in
    this design (see ``WITHIN_SUBJECT_PLAN.md``), so reusing "validation"
    would misleadingly imply one exists.
    """

    train_graph_indices: np.ndarray
    evaluation_graph_indices: np.ndarray
    subject_id: int
    fold: int


def _validate_within_subject_folds(
    folds: tuple[WithinSubjectFold, ...],
    subject_ids: np.ndarray,
    target_subject: int,
) -> None:
    target_graph_indices = set(
        map(int, np.flatnonzero(subject_ids == target_subject))
    )

    seen_in_evaluation: set[int] = set()
    for fold in folds:
        if fold.subject_id != target_subject:
            raise RuntimeError(
                f"Within-subject fold {fold.fold} is tagged with the wrong subject_id"
            )
        train_set = set(map(int, fold.train_graph_indices))
        evaluation_set = set(map(int, fold.evaluation_graph_indices))
        if train_set & evaluation_set:
            raise RuntimeError(
                f"Within-subject fold {fold.fold} has overlapping graph indices"
            )
        if seen_in_evaluation & evaluation_set:
            raise RuntimeError(
                "Within-subject folds assign a trial to evaluation more than once"
            )
        seen_in_evaluation |= evaluation_set
        if (train_set | evaluation_set) != target_graph_indices:
            raise RuntimeError(
                f"Within-subject fold {fold.fold} does not exactly cover subject "
                f"{target_subject}'s own trials"
            )

    if seen_in_evaluation != target_graph_indices:
        raise RuntimeError(
            "Within-subject folds do not cover every one of the subject's trials "
            "exactly once across evaluation sets"
        )


def create_within_subject_folds(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    target_subject: int,
    *,
    k: int = 5,
    seed: int = 42,
) -> tuple[WithinSubjectFold, ...]:
    """Create ``k`` reproducible, class-stratified folds over one subject's trials.

    Every fold's ``evaluation_graph_indices`` is disjoint and, together
    across all ``k`` folds, covers every one of ``target_subject``'s trials
    exactly once, with class balance preserved in each evaluation fold via
    stratification. There is no early-stopping validation split here (see
    ``WITHIN_SUBJECT_PLAN.md``) -- each fold's ``train_graph_indices`` is
    simply everything of the subject's not in that fold's evaluation set.
    """

    raw_subject_ids = np.asarray(subject_ids)
    if raw_subject_ids.ndim != 1 or len(raw_subject_ids) == 0:
        raise ValueError("subject_ids must be a non-empty one-dimensional array")
    if not np.issubdtype(raw_subject_ids.dtype, np.integer):
        raise TypeError("subject_ids must have an integer dtype")
    subject_values = raw_subject_ids.astype(np.int64, copy=False)

    raw_labels = np.asarray(labels)
    if raw_labels.shape != subject_values.shape:
        raise ValueError("labels must be the same shape as subject_ids")

    resolved_target_subject = int(target_subject)
    if resolved_target_subject not in set(map(int, np.unique(subject_values))):
        raise ValueError(
            f"target_subject {resolved_target_subject} is absent from subject_ids"
        )

    target_graph_indices = _graph_indices_for_subjects(
        subject_values, (resolved_target_subject,)
    )
    target_labels = raw_labels[target_graph_indices]

    if k < 2:
        raise ValueError("k must be at least 2")
    if k > len(target_graph_indices):
        raise ValueError(
            f"k={k} folds requires at least {k} trials for subject "
            f"{resolved_target_subject}; got {len(target_graph_indices)}"
        )

    stratified_k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds: list[WithinSubjectFold] = []
    for fold, (train_positions, evaluation_positions) in enumerate(
        stratified_k_fold.split(target_graph_indices, target_labels)
    ):
        folds.append(
            WithinSubjectFold(
                train_graph_indices=np.sort(target_graph_indices[train_positions]),
                evaluation_graph_indices=np.sort(
                    target_graph_indices[evaluation_positions]
                ),
                subject_id=resolved_target_subject,
                fold=fold,
            )
        )

    result = tuple(folds)
    _validate_within_subject_folds(result, subject_values, resolved_target_subject)
    return result
