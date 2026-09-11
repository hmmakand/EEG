"""Tests for within-subject split construction.

Uses synthetic subject-ID/label arrays only -- no combination or the real
dataset is involved, mirroring ``test_loso_split.py``'s approach for the
LOSO/inner-CV split types.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.ourexperimentversionfour.data.within_subject_split import (
    WithinSubjectFold,
    _validate_within_subject_folds,
    create_within_subject_folds,
)


def _synthetic_subject_ids_and_labels(
    subjects: int = 5, trials_per_subject: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    subject_ids = np.repeat(np.arange(1, subjects + 1), trials_per_subject).astype(
        np.int64
    )
    half = trials_per_subject // 2
    labels = np.tile(
        np.array([0] * half + [1] * (trials_per_subject - half), dtype=np.int64),
        subjects,
    )
    return subject_ids, labels


class CreateWithinSubjectFoldsTests(unittest.TestCase):
    def test_partition_covers_every_trial_exactly_once(self) -> None:
        subject_ids, labels = _synthetic_subject_ids_and_labels()
        folds = create_within_subject_folds(subject_ids, labels, 3, k=5, seed=1)

        self.assertEqual(len(folds), 5)
        target_indices = set(map(int, np.flatnonzero(subject_ids == 3)))
        seen: set[int] = set()
        for fold in folds:
            evaluation_set = set(map(int, fold.evaluation_graph_indices))
            self.assertFalse(seen & evaluation_set)
            seen |= evaluation_set
        self.assertEqual(seen, target_indices)

    def test_train_and_evaluation_disjoint_per_fold(self) -> None:
        subject_ids, labels = _synthetic_subject_ids_and_labels()
        folds = create_within_subject_folds(subject_ids, labels, 2, k=5, seed=1)
        for fold in folds:
            train_set = set(map(int, fold.train_graph_indices))
            evaluation_set = set(map(int, fold.evaluation_graph_indices))
            self.assertFalse(train_set & evaluation_set)

    def test_evaluation_folds_stay_class_balanced(self) -> None:
        subject_ids, labels = _synthetic_subject_ids_and_labels(
            subjects=4, trials_per_subject=20
        )
        folds = create_within_subject_folds(subject_ids, labels, 1, k=5, seed=7)
        for fold in folds:
            fold_labels = labels[fold.evaluation_graph_indices]
            self.assertEqual(int((fold_labels == 0).sum()), 2)
            self.assertEqual(int((fold_labels == 1).sum()), 2)

    def test_no_other_subjects_indices_appear(self) -> None:
        subject_ids, labels = _synthetic_subject_ids_and_labels()
        folds = create_within_subject_folds(subject_ids, labels, 4, k=5, seed=1)
        for fold in folds:
            self.assertTrue(
                np.all(subject_ids[fold.train_graph_indices] == 4)
            )
            self.assertTrue(
                np.all(subject_ids[fold.evaluation_graph_indices] == 4)
            )
            self.assertEqual(fold.subject_id, 4)

    def test_reproducible_given_same_seed(self) -> None:
        subject_ids, labels = _synthetic_subject_ids_and_labels()
        first = create_within_subject_folds(subject_ids, labels, 1, k=5, seed=42)
        repeated = create_within_subject_folds(subject_ids, labels, 1, k=5, seed=42)
        for left, right in zip(first, repeated):
            self.assertTrue(
                np.array_equal(
                    left.evaluation_graph_indices, right.evaluation_graph_indices
                )
            )

    def test_different_seeds_reshuffle_fold_membership(self) -> None:
        subject_ids, labels = _synthetic_subject_ids_and_labels()
        first = create_within_subject_folds(subject_ids, labels, 1, k=5, seed=1)
        second = create_within_subject_folds(subject_ids, labels, 1, k=5, seed=2)
        self.assertFalse(
            all(
                np.array_equal(
                    left.evaluation_graph_indices, right.evaluation_graph_indices
                )
                for left, right in zip(first, second)
            )
        )

    def test_rejects_k_larger_than_subject_trial_count(self) -> None:
        subject_ids, labels = _synthetic_subject_ids_and_labels(
            subjects=2, trials_per_subject=4
        )
        with self.assertRaisesRegex(ValueError, "requires at least"):
            create_within_subject_folds(subject_ids, labels, 1, k=10)

    def test_rejects_target_subject_absent_from_subject_ids(self) -> None:
        subject_ids, labels = _synthetic_subject_ids_and_labels()
        with self.assertRaisesRegex(ValueError, "absent from subject_ids"):
            create_within_subject_folds(subject_ids, labels, 99, k=5)

    def test_rejects_non_integer_subject_ids(self) -> None:
        with self.assertRaises(TypeError):
            create_within_subject_folds(
                np.asarray([1.0, 2.0, 3.0]), np.asarray([0, 1, 0]), 1
            )

    def test_rejects_mismatched_labels_shape(self) -> None:
        subject_ids, _labels = _synthetic_subject_ids_and_labels()
        with self.assertRaisesRegex(ValueError, "same shape"):
            create_within_subject_folds(subject_ids, np.asarray([0, 1]), 1, k=5)

    def test_rejects_k_below_two(self) -> None:
        subject_ids, labels = _synthetic_subject_ids_and_labels()
        with self.assertRaisesRegex(ValueError, "at least 2"):
            create_within_subject_folds(subject_ids, labels, 1, k=1)


class ValidateWithinSubjectFoldsTests(unittest.TestCase):
    def test_detects_overlapping_graph_indices(self) -> None:
        subject_ids, _labels = _synthetic_subject_ids_and_labels(
            subjects=2, trials_per_subject=4
        )
        bad_folds = (
            WithinSubjectFold(
                train_graph_indices=np.array([0, 1]),
                evaluation_graph_indices=np.array([1, 2, 3]),
                subject_id=1,
                fold=0,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "overlapping graph indices"):
            _validate_within_subject_folds(bad_folds, subject_ids, 1)

    def test_detects_a_trial_evaluated_more_than_once(self) -> None:
        subject_ids, _labels = _synthetic_subject_ids_and_labels(
            subjects=2, trials_per_subject=4
        )
        bad_folds = (
            WithinSubjectFold(
                train_graph_indices=np.array([2, 3]),
                evaluation_graph_indices=np.array([0, 1]),
                subject_id=1,
                fold=0,
            ),
            WithinSubjectFold(
                train_graph_indices=np.array([2, 3]),
                evaluation_graph_indices=np.array([0, 1]),
                subject_id=1,
                fold=1,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            _validate_within_subject_folds(bad_folds, subject_ids, 1)

    def test_detects_another_subjects_indices(self) -> None:
        subject_ids, _labels = _synthetic_subject_ids_and_labels(
            subjects=2, trials_per_subject=4
        )
        bad_folds = (
            WithinSubjectFold(
                train_graph_indices=np.array([0, 1, 4]),  # 4 belongs to subject 2
                evaluation_graph_indices=np.array([2, 3]),
                subject_id=1,
                fold=0,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "does not exactly cover"):
            _validate_within_subject_folds(bad_folds, subject_ids, 1)


if __name__ == "__main__":
    unittest.main()
