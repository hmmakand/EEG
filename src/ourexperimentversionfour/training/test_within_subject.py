"""Integration tests for the within-subject classification diagnostic."""

from __future__ import annotations

import math
import unittest

from src.ourexperimentversionfour.data import load_dataset
from src.ourexperimentversionfour.data.combinations import get_combination
from src.ourexperimentversionfour.training.config import WithinSubjectConfig
from src.ourexperimentversionfour.training.within_subject import (
    WithinSubjectFoldResult,
    summarize_within_subject_results,
    train_within_subject,
)
from src.ourexperimentversionfour.training.within_subject_cli import build_parser


class TrainWithinSubjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.combination = get_combination("without_csd_alpha_wpli")
        cls.dataset = load_dataset()

    def test_completes_and_returns_expected_number_of_results(self) -> None:
        config = WithinSubjectConfig(folds=2, epochs=2, save_outputs=False)
        results = train_within_subject(
            1,
            config,
            dataset=self.dataset,
            combination=self.combination,
            show_progress=False,
        )
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, WithinSubjectFoldResult)
            self.assertEqual(result.subject_id, 1)
            self.assertEqual(result.repeat, 0)

    def test_every_metric_is_finite_and_bounded(self) -> None:
        config = WithinSubjectConfig(folds=2, epochs=2, save_outputs=False)
        results = train_within_subject(
            2,
            config,
            dataset=self.dataset,
            combination=self.combination,
            show_progress=False,
        )
        for result in results:
            metrics = result.evaluation.as_dict()
            for name, value in metrics.items():
                if name == "examples":
                    continue
                self.assertTrue(math.isfinite(value), f"{name}={value} is not finite")
            self.assertGreaterEqual(metrics["accuracy"], 0.0)
            self.assertLessEqual(metrics["accuracy"], 1.0)
            self.assertGreaterEqual(metrics["balanced_accuracy"], 0.0)
            self.assertLessEqual(metrics["balanced_accuracy"], 1.0)
            self.assertGreaterEqual(metrics["auc"], 0.0)
            self.assertLessEqual(metrics["auc"], 1.0)

    def test_same_seed_is_reproducible(self) -> None:
        config = WithinSubjectConfig(folds=2, epochs=2, save_outputs=False)
        first = train_within_subject(
            3,
            config,
            dataset=self.dataset,
            combination=self.combination,
            show_progress=False,
        )
        second = train_within_subject(
            3,
            config,
            dataset=self.dataset,
            combination=self.combination,
            show_progress=False,
        )
        for left, right in zip(first, second):
            self.assertEqual(left.as_dict(), right.as_dict())

    def test_repeats_produce_more_results_with_different_fold_membership(
        self,
    ) -> None:
        config = WithinSubjectConfig(folds=2, epochs=2, repeats=2, save_outputs=False)
        results = train_within_subject(
            4,
            config,
            dataset=self.dataset,
            combination=self.combination,
            show_progress=False,
        )
        self.assertEqual(len(results), 4)
        self.assertEqual({result.repeat for result in results}, {0, 1})

        first_repeat_fold_zero = next(
            result for result in results if result.repeat == 0 and result.fold == 0
        )
        second_repeat_fold_zero = next(
            result for result in results if result.repeat == 1 and result.fold == 0
        )
        # Different seeds per repeat reshuffle fold membership, so the two
        # repeats' fold-0 results should not be identical.
        self.assertNotEqual(
            first_repeat_fold_zero.as_dict(), second_repeat_fold_zero.as_dict()
        )

    def test_can_select_sgd_instead_of_adamw(self) -> None:
        config = WithinSubjectConfig(
            folds=2, epochs=2, optimizer="sgd", momentum=0.9, save_outputs=False
        )
        results = train_within_subject(
            5,
            config,
            dataset=self.dataset,
            combination=self.combination,
            show_progress=False,
        )
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(math.isfinite(result.evaluation.loss))


class SummarizeWithinSubjectResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.combination = get_combination("without_csd_alpha_wpli")
        cls.dataset = load_dataset()

    def test_aggregates_with_equal_subject_weighting(self) -> None:
        config = WithinSubjectConfig(folds=2, epochs=2, save_outputs=False)
        results = []
        for subject_id in (1, 2):
            results.extend(
                train_within_subject(
                    subject_id,
                    config,
                    dataset=self.dataset,
                    combination=self.combination,
                    show_progress=False,
                )
            )
        summary = summarize_within_subject_results(results)
        self.assertEqual(summary["subjects"], 2)
        self.assertEqual(summary["total_folds"], 4)
        self.assertIn("1", summary["per_subject"])
        self.assertIn("2", summary["per_subject"])
        self.assertGreaterEqual(summary["mean_balanced_accuracy"], 0.0)
        self.assertLessEqual(summary["mean_balanced_accuracy"], 1.0)

    def test_rejects_empty_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one fold result"):
            summarize_within_subject_results([])


class WithinSubjectConfigTests(unittest.TestCase):
    def test_rejects_invalid_optimizer_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer must be"):
            WithinSubjectConfig(optimizer="rmsprop")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "momentum must be"):
            WithinSubjectConfig(momentum=1.5)

    def test_cli_exposes_optimizer_and_momentum_flags(self) -> None:
        parser = build_parser()
        option_destinations = {
            action.dest for action in parser._actions  # type: ignore[attr-defined]
        }
        self.assertIn("optimizer", option_destinations)
        self.assertIn("momentum", option_destinations)


if __name__ == "__main__":
    unittest.main()
