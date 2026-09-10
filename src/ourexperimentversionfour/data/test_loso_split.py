"""Tests for combination-agnostic LOSO split construction.

Uses synthetic subject-ID arrays only -- no combination or the real dataset
is involved, since this module has no dependency on either.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.ourexperimentversionfour.data.loso_split import (
    GraphDataLoaderConfig,
    InnerCvFold,
    LosoGraphSplit,
    _validate_inner_folds,
    _validate_split,
    create_inner_cv_folds,
    create_loso_splits,
)


def _synthetic_subject_ids(
    subjects: int = 10, trials_per_subject: int = 4
) -> np.ndarray:
    return np.repeat(np.arange(1, subjects + 1), trials_per_subject).astype(np.int64)


class CreateLosoSplitsTests(unittest.TestCase):
    def test_one_fold_per_subject_reproducible_and_complete(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=10, trials_per_subject=4)
        first = create_loso_splits(subject_ids, validation_subjects=2, seed=7)
        repeated = create_loso_splits(subject_ids, validation_subjects=2, seed=7)

        self.assertEqual(len(first), 10)
        self.assertEqual(
            tuple(split.test_subject_id for split in first), tuple(range(1, 11))
        )
        for split, same_split in zip(first, repeated):
            self.assertEqual(len(split.train_subject_ids), 7)
            self.assertEqual(len(split.validation_subject_ids), 2)
            self.assertEqual(len(split.train_graph_indices), 7 * 4)
            self.assertEqual(len(split.validation_graph_indices), 2 * 4)
            self.assertEqual(len(split.test_graph_indices), 4)
            self.assertEqual(
                split.validation_subject_ids, same_split.validation_subject_ids
            )
            self.assertTrue(
                np.array_equal(
                    split.train_graph_indices, same_split.train_graph_indices
                )
            )

    def test_subjects_never_appear_in_more_than_one_partition(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=10, trials_per_subject=4)
        for split in create_loso_splits(subject_ids, validation_subjects=2, seed=1):
            subject_sets = (
                set(split.train_subject_ids),
                set(split.validation_subject_ids),
                {split.test_subject_id},
            )
            self.assertFalse(subject_sets[0] & subject_sets[1])
            self.assertFalse(subject_sets[0] & subject_sets[2])
            self.assertFalse(subject_sets[1] & subject_sets[2])
            # And union covers every subject exactly once.
            self.assertEqual(
                subject_sets[0] | subject_sets[1] | subject_sets[2],
                set(range(1, 11)),
            )

    def test_different_seeds_can_change_the_validation_draw(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=10, trials_per_subject=4)
        first = create_loso_splits(subject_ids, validation_subjects=2, seed=1)
        second = create_loso_splits(subject_ids, validation_subjects=2, seed=2)
        self.assertNotEqual(
            tuple(split.validation_subject_ids for split in first),
            tuple(split.validation_subject_ids for split in second),
        )

    def test_rejects_too_few_subjects(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=2, trials_per_subject=4)
        with self.assertRaisesRegex(ValueError, "[Aa]t least three subjects"):
            create_loso_splits(subject_ids)

    def test_rejects_validation_subjects_leaving_no_training_subject(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=3, trials_per_subject=4)
        with self.assertRaisesRegex(ValueError, "validation_subjects must leave"):
            create_loso_splits(subject_ids, validation_subjects=2)

    def test_rejects_non_integer_subject_ids(self) -> None:
        with self.assertRaises(TypeError):
            create_loso_splits(np.asarray([1.0, 2.0, 3.0]))

    def test_rejects_empty_subject_ids(self) -> None:
        with self.assertRaises(ValueError):
            create_loso_splits(np.asarray([], dtype=np.int64))


class ValidateSplitTests(unittest.TestCase):
    def test_detects_overlapping_graph_indices(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=5, trials_per_subject=2)
        bad_split = LosoGraphSplit(
            train_graph_indices=np.array([0, 1, 2]),
            validation_graph_indices=np.array([2, 3]),  # overlaps with train
            test_graph_indices=np.array([4, 5]),
            train_subject_ids=(1, 2),
            validation_subject_ids=(3,),
            test_subject_id=4,
            fold=0,
        )
        with self.assertRaisesRegex(RuntimeError, "overlapping graph indices"):
            _validate_split(bad_split, subject_ids)

    def test_detects_subject_leakage_across_partitions(self) -> None:
        # 5 subjects x 2 trials = 10 graphs (indices 0-9). Indices are fully
        # disjoint and cover every graph, so only the *declared* subject-ID
        # tuples overlap (subject 2 claimed by both train and validation) --
        # isolating the subject-leakage check from the graph-coverage checks.
        subject_ids = _synthetic_subject_ids(subjects=5, trials_per_subject=2)
        bad_split = LosoGraphSplit(
            train_graph_indices=np.array([0, 1, 2, 3]),
            validation_graph_indices=np.array([4, 5, 6, 7]),
            test_graph_indices=np.array([8, 9]),
            train_subject_ids=(1, 2),
            validation_subject_ids=(2, 4),  # subject 2 leaks into validation too
            test_subject_id=5,
            fold=0,
        )
        with self.assertRaisesRegex(RuntimeError, "leak subjects"):
            _validate_split(bad_split, subject_ids)


class CreateInnerCvFoldsTests(unittest.TestCase):
    def test_partitions_every_eligible_subject_exactly_once(self) -> None:
        # 20 subjects, hold one out as the "outer test subject" (not eligible).
        subject_ids = _synthetic_subject_ids(subjects=20, trials_per_subject=4)
        eligible = tuple(s for s in range(1, 21) if s != 1)  # 19 development subjects
        folds = create_inner_cv_folds(subject_ids, eligible, k=4, seed=42)

        self.assertEqual(len(folds), 4)
        seen_in_validation: set[int] = set()
        for fold in folds:
            train_set = set(fold.train_subject_ids)
            validation_set = set(fold.validation_subject_ids)
            self.assertFalse(train_set & validation_set)
            self.assertEqual(train_set | validation_set, set(eligible))
            self.assertFalse(seen_in_validation & validation_set)
            seen_in_validation |= validation_set
            # Subject 1 (the outer test subject) never appears anywhere.
            self.assertNotIn(1, train_set)
            self.assertNotIn(1, validation_set)
        self.assertEqual(seen_in_validation, set(eligible))

    def test_uneven_division_still_balances_within_one(self) -> None:
        # 9 eligible subjects over k=4 folds: sizes must be 3,2,2,2 (or any
        # split differing by at most 1), not e.g. 6,1,1,1.
        subject_ids = _synthetic_subject_ids(subjects=9, trials_per_subject=3)
        eligible = tuple(range(1, 10))
        folds = create_inner_cv_folds(subject_ids, eligible, k=4, seed=1)
        validation_sizes = sorted(len(fold.validation_subject_ids) for fold in folds)
        self.assertEqual(validation_sizes, [2, 2, 2, 3])

    def test_reproducible_given_same_seed(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=20, trials_per_subject=4)
        eligible = tuple(range(1, 20))
        first = create_inner_cv_folds(subject_ids, eligible, k=5, seed=7)
        repeated = create_inner_cv_folds(subject_ids, eligible, k=5, seed=7)
        self.assertEqual(
            tuple(f.validation_subject_ids for f in first),
            tuple(f.validation_subject_ids for f in repeated),
        )

    def test_different_seeds_change_fold_membership(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=20, trials_per_subject=4)
        eligible = tuple(range(1, 20))
        first = create_inner_cv_folds(subject_ids, eligible, k=5, seed=1)
        second = create_inner_cv_folds(subject_ids, eligible, k=5, seed=2)
        self.assertNotEqual(
            tuple(f.validation_subject_ids for f in first),
            tuple(f.validation_subject_ids for f in second),
        )

    def test_rejects_k_larger_than_eligible_subject_count(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=5, trials_per_subject=4)
        with self.assertRaisesRegex(ValueError, "requires at least"):
            create_inner_cv_folds(subject_ids, tuple(range(1, 6)), k=10)

    def test_rejects_duplicate_eligible_subjects(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=5, trials_per_subject=4)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            create_inner_cv_folds(subject_ids, (1, 2, 2, 3, 4), k=2)

    def test_rejects_eligible_subject_absent_from_data(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=5, trials_per_subject=4)
        with self.assertRaisesRegex(ValueError, "absent from subject_ids"):
            create_inner_cv_folds(subject_ids, (1, 2, 3, 99), k=2)


class ValidateInnerFoldsTests(unittest.TestCase):
    def test_detects_a_subject_validated_more_than_once(self) -> None:
        subject_ids = _synthetic_subject_ids(subjects=4, trials_per_subject=2)
        bad_folds = (
            InnerCvFold(
                train_graph_indices=np.array([4, 5, 6, 7]),
                validation_graph_indices=np.array([0, 1]),
                train_subject_ids=(3, 4),
                validation_subject_ids=(1,),
                fold=0,
            ),
            InnerCvFold(
                train_graph_indices=np.array([0, 1, 6, 7]),
                validation_graph_indices=np.array([2, 3]),
                train_subject_ids=(1, 4),
                validation_subject_ids=(2,),
                fold=1,
            ),
            InnerCvFold(
                train_graph_indices=np.array([0, 1, 2, 3]),
                validation_graph_indices=np.array([2, 3]),
                train_subject_ids=(1, 2),
                validation_subject_ids=(2,),  # subject 2 validated twice
                fold=2,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            _validate_inner_folds(bad_folds, subject_ids, (1, 2, 3, 4))


class GraphDataLoaderConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        config = GraphDataLoaderConfig()
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.validation_subjects, 5)

    def test_invalid_configuration_is_rejected(self) -> None:
        for keyword_arguments in (
            {"batch_size": 0},
            {"validation_subjects": 0},
            {"num_workers": -1},
            {"persistent_workers": True, "num_workers": 0},
            {"normalization_epsilon": 0},
        ):
            with self.subTest(keyword_arguments=keyword_arguments):
                with self.assertRaises(ValueError):
                    GraphDataLoaderConfig(**keyword_arguments)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
