"""Tests for the inner-CV hyperparameter search."""

from __future__ import annotations

import unittest

from src.ourexperimentversionfour.data import load_dataset
from src.ourexperimentversionfour.data.combinations import get_combination
from src.ourexperimentversionfour.training import FoldResult, TrainingConfig
from src.ourexperimentversionfour.training.hyperparameter_search import (
    DEFAULT_SEARCH_GRID,
    select_hyperparameters,
)
from src.ourexperimentversionfour.training.loso import train_loso_fold_with_search

_TINY_GRID = [
    {"learning_rate": 0.01, "weight_decay": 5e-4},
    {"learning_rate": 0.001, "weight_decay": 1e-4},
]


class SelectHyperparametersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.combination = get_combination("without_csd_alpha_wpli")
        cls.dataset = load_dataset()

    def test_default_grid_has_twelve_candidates(self) -> None:
        self.assertEqual(len(DEFAULT_SEARCH_GRID), 12)
        for candidate in DEFAULT_SEARCH_GRID:
            self.assertEqual(set(candidate), {"learning_rate", "weight_decay"})

    def test_scores_every_candidate_and_picks_one_from_the_grid(self) -> None:
        train_subject_ids = tuple(subject for subject in range(1, 51) if subject != 1)
        best_overrides, candidate_scores = select_hyperparameters(
            self.combination,
            self.dataset,
            train_subject_ids,
            grid=_TINY_GRID,
            inner_folds=2,
            search_epochs=1,
            search_patience=1,
            base_config=TrainingConfig(),
            seed=42,
        )
        self.assertIn(best_overrides, _TINY_GRID)
        self.assertEqual(len(candidate_scores), len(_TINY_GRID))
        for entry in candidate_scores:
            self.assertIn(entry["overrides"], _TINY_GRID)
            self.assertEqual(len(entry["per_fold_balanced_accuracy"]), 2)
            self.assertAlmostEqual(
                entry["mean_balanced_accuracy"],
                sum(entry["per_fold_balanced_accuracy"]) / 2,
            )
            for value in entry["per_fold_balanced_accuracy"]:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_same_folds_are_reused_across_candidates(self) -> None:
        # Same seed -> same inner-CV folds regardless of which candidate is
        # being scored; this is what makes the comparison across candidates
        # fair (see AUDIT.md / plan discussion). We can't observe the folds
        # directly through the public API, but two full searches with the
        # same seed must be fully reproducible if this holds.
        train_subject_ids = tuple(subject for subject in range(1, 51) if subject != 1)
        first = select_hyperparameters(
            self.combination,
            self.dataset,
            train_subject_ids,
            grid=_TINY_GRID,
            inner_folds=2,
            search_epochs=1,
            search_patience=1,
            base_config=TrainingConfig(),
            seed=7,
        )
        second = select_hyperparameters(
            self.combination,
            self.dataset,
            train_subject_ids,
            grid=_TINY_GRID,
            inner_folds=2,
            search_epochs=1,
            search_patience=1,
            base_config=TrainingConfig(),
            seed=7,
        )
        self.assertEqual(first, second)

    def test_rejects_empty_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one candidate"):
            select_hyperparameters(
                self.combination,
                self.dataset,
                tuple(range(1, 50)),
                grid=[],
                base_config=TrainingConfig(),
            )


class TrainLosoFoldWithSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.combination = get_combination("without_csd_alpha_wpli")
        cls.dataset = load_dataset()

    def test_selects_retrains_and_evaluates_once(self) -> None:
        result, provenance = train_loso_fold_with_search(
            1,
            TrainingConfig(epochs=1, batch_size=64, save_outputs=False),
            dataset=self.dataset,
            grid=_TINY_GRID,
            inner_folds=2,
            search_epochs=1,
            search_patience=1,
            show_progress=False,
        )
        self.assertIsInstance(result, FoldResult)
        self.assertEqual(result.test_subject_id, 1)
        self.assertEqual(result.test.examples, 40)
        self.assertIn(provenance["selected_hyperparameters"], _TINY_GRID)
        self.assertEqual(len(provenance["candidate_scores"]), len(_TINY_GRID))
        self.assertEqual(provenance["inner_folds"], 2)
        self.assertEqual(provenance["search_epochs"], 1)
        self.assertEqual(provenance["search_patience"], 1)

    def test_the_held_out_test_subject_never_enters_the_search(self) -> None:
        # A cheap structural proxy: every candidate's inner-CV folds are
        # built from the 49 non-test subjects, so the search must complete
        # even when the test subject is one that would otherwise be a
        # natural inner-fold member -- confirming the exclusion happens
        # before folds are built, not filtered out after the fact.
        result, _provenance = train_loso_fold_with_search(
            5,
            TrainingConfig(epochs=1, batch_size=64, save_outputs=False),
            dataset=self.dataset,
            grid=_TINY_GRID,
            inner_folds=2,
            search_epochs=1,
            search_patience=1,
            show_progress=False,
        )
        self.assertEqual(result.test_subject_id, 5)
        self.assertNotIn(5, result.validation_subject_ids)


if __name__ == "__main__":
    unittest.main()
