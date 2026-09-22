"""Integration tests for the within-subject classification diagnostic."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from src.ourexperimentversionfive.training.engine import evaluate

from src.ourexperimentversionfive.data import load_dataset
from src.ourexperimentversionfive.data.combinations import get_combination
from src.ourexperimentversionfive.training.config import WithinSubjectConfig
from src.ourexperimentversionfive.training.within_subject import (
    WithinSubjectFoldResult,
    summarize_within_subject_results,
    train_within_subject,
)
from src.ourexperimentversionfive.training.within_subject_cli import build_parser


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

    def test_epoch_count_is_selected_on_inner_validation_not_training_loss(
        self,
    ) -> None:
        """Regression test for the training-loss-checkpointing overfitting bug.

        ``best_epoch``/``epochs_ran`` must come from the inner-validation
        selection pass, not from running the full ``epochs`` budget and
        picking the lowest training loss (which, pre-fix, was almost always
        the last epoch).
        """

        config = WithinSubjectConfig(
            folds=2, epochs=6, patience=1, save_outputs=False
        )
        results = train_within_subject(
            6,
            config,
            dataset=self.dataset,
            combination=self.combination,
            show_progress=False,
        )
        for result in results:
            self.assertIsNotNone(result.inner_selection)
            self.assertEqual(result.best_epoch, result.inner_selection["selected_epoch"])
            self.assertEqual(result.epochs_ran, result.best_epoch)
            self.assertLessEqual(result.best_epoch, config.epochs)
            self.assertGreaterEqual(result.best_epoch, 1)
            self.assertEqual(len(result.history), result.best_epoch)

    def test_diagnostics_use_frozen_final_model_and_preserve_selection(self) -> None:
        calls = []

        def checked_evaluate(model, loader, loss_function, device):
            before = {key: value.clone() for key, value in model.state_dict().items()}
            metrics = evaluate(model, loader, loss_function, device)
            self.assertFalse(model.training)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]), key)
            calls.append((model, before, metrics))
            return metrics

        config = WithinSubjectConfig(folds=2, epochs=2, node_normalization="zscore",
                                     device="cpu", save_outputs=False)
        with patch("src.ourexperimentversionfive.training.within_subject.evaluate",
                   side_effect=checked_evaluate):
            results = train_within_subject(1, config, dataset=self.dataset,
                                          combination=self.combination, show_progress=False)
        offset = 0
        for result in results:
            inner = result.inner_selection
            history = inner["history"]
            self.assertEqual(len(history), inner["epochs_ran"])
            self.assertEqual(history[result.best_epoch - 1]["validation"]["loss"],
                             inner["best_validation_loss"])
            train_indices = set(inner["inner_train_graph_indices"])
            val_indices = set(inner["inner_validation_graph_indices"])
            self.assertFalse(train_indices & val_indices)
            self.assertEqual(train_indices | val_indices,
                             set(result.preprocessing["train_graph_indices"]))
            self.assertFalse((train_indices | val_indices) &
                             set(result.preprocessing["evaluation_graph_indices"]))
            for record in history:
                self.assertEqual(record["training_online"]["examples"], len(train_indices))
                self.assertEqual(record["validation"]["examples"], len(val_indices))
            outer_call = calls[offset + len(history)]
            train_call = calls[offset + len(history) + 1]
            self.assertIs(outer_call[0], train_call[0])
            for key in outer_call[1]:
                self.assertTrue(torch.equal(outer_call[1][key], train_call[1][key]))
            self.assertEqual(result.final_training, train_call[2])
            self.assertEqual(result.final_training.examples,
                             len(result.preprocessing["train_graph_indices"]))
            saved = result.as_dict()
            self.assertAlmostEqual(saved["generalization_gap"]["loss"],
                                   result.evaluation.loss - result.final_training.loss)
            self.assertAlmostEqual(saved["generalization_gap"]["balanced_accuracy"],
                                   result.final_training.balanced_accuracy - result.evaluation.balanced_accuracy)
            offset += len(history) + 2
        summary = summarize_within_subject_results(list(results))
        self.assertAlmostEqual(summary["diagnostics"]["mean_gap_balanced_accuracy"],
                               sum(r.generalization_gap()["balanced_accuracy"] for r in results) / 2)

    def test_default_batch_size_gives_multiple_batches_per_fold(self) -> None:
        """Regression test: at the old default (32), every fold's training
        set (24-32 trials) fit in one batch, so every "epoch" was a single
        full-batch gradient step -- which, combined with Adam's unstable
        early moment estimates, caused >40% of folds to early-stop after
        just one step on pure noise (final train accuracy at or below
        chance). The default must stay well below the smallest training
        split (the inner-validation split's training side, 75% of a 32-trial
        fold by default) so each epoch is several real mini-batch updates.
        """

        config = WithinSubjectConfig()
        smallest_realistic_training_split = 24  # 75% of a 32-trial outer fold
        self.assertLess(config.batch_size, smallest_realistic_training_split)

    def test_inner_selection_exactly_partitions_the_training_trials(self) -> None:
        """The inner split's example counts must sum to the fold's training
        count exactly -- if it ever summed to more, or included the fold's
        evaluation trials, this would catch it via a count mismatch."""

        config = WithinSubjectConfig(folds=2, epochs=3, save_outputs=False)
        results = train_within_subject(
            7,
            config,
            dataset=self.dataset,
            combination=self.combination,
            show_progress=False,
        )
        for result in results:
            inner_examples = (
                result.inner_selection["inner_train_examples"]
                + result.inner_selection["inner_validation_examples"]
            )
            train_examples = len(result.preprocessing["train_graph_indices"])
            self.assertEqual(inner_examples, train_examples)


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

    def test_rejects_invalid_patience_and_minimum_improvement(self) -> None:
        with self.assertRaisesRegex(ValueError, "patience must be"):
            WithinSubjectConfig(patience=0)
        with self.assertRaisesRegex(ValueError, "minimum_improvement must be"):
            WithinSubjectConfig(minimum_improvement=-1e-4)

    def test_allows_disabling_early_stopping(self) -> None:
        config = WithinSubjectConfig(patience=None)
        self.assertIsNone(config.patience)

    def test_rejects_invalid_inner_validation_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "inner_validation_fraction must be"):
            WithinSubjectConfig(inner_validation_fraction=0.0)
        with self.assertRaisesRegex(ValueError, "inner_validation_fraction must be"):
            WithinSubjectConfig(inner_validation_fraction=1.0)

    def test_cli_exposes_optimizer_and_momentum_flags(self) -> None:
        parser = build_parser()
        option_destinations = {
            action.dest for action in parser._actions  # type: ignore[attr-defined]
        }
        self.assertIn("optimizer", option_destinations)
        self.assertIn("momentum", option_destinations)

    def test_cli_exposes_patience_and_inner_validation_flags(self) -> None:
        parser = build_parser()
        option_destinations = {
            action.dest for action in parser._actions  # type: ignore[attr-defined]
        }
        self.assertIn("patience", option_destinations)
        self.assertIn("minimum_improvement", option_destinations)
        self.assertIn("inner_validation_fraction", option_destinations)


if __name__ == "__main__":
    unittest.main()
