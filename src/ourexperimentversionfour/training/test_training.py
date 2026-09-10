"""Tests for the without-CSD alpha-band wPLI training workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PygDataLoader

from src.ourexperimentversionfour.model import EEGGCN1, EEGGCN1Config
from src.ourexperimentversionfour.training import (
    ClassificationMetrics,
    FoldResult,
    TrainingConfig,
    evaluate,
    summarize_results,
    train_epoch,
    train_loso_fold,
)
from src.ourexperimentversionfour.training.metrics import calculate_metrics
from src.ourexperimentversionfour.training.train import build_parser


def _ring_edge_index(nodes: int) -> torch.Tensor:
    """Build a directed ring (each node connected to its two neighbors)."""

    sources = torch.cat([torch.arange(nodes), torch.arange(nodes)])
    targets = torch.cat(
        [torch.roll(torch.arange(nodes), shifts=-1), torch.roll(torch.arange(nodes), shifts=1)]
    )
    return torch.stack([sources, targets])


def _graphs() -> PygDataLoader:
    """Build 8 deterministic sparse graphs for engine-level tests."""

    generator = torch.Generator().manual_seed(42)
    edge_index = _ring_edge_index(29)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    graphs = [
        Data(
            x=torch.randn(29, 6, generator=generator) + graph_index % 2,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(-1),
            y=torch.tensor([graph_index % 2], dtype=torch.long),
        )
        for graph_index in range(8)
    ]
    return PygDataLoader(graphs, batch_size=4, shuffle=False)


class TrainingUtilityTests(unittest.TestCase):
    def test_metrics_are_correct(self) -> None:
        metrics = calculate_metrics(
            total_loss=2.0,
            labels=np.asarray([0, 0, 1, 1]),
            predictions=np.asarray([0, 1, 1, 1]),
            scores=np.asarray([0.0, 1.0, 1.0, 1.0]),
        )
        self.assertAlmostEqual(metrics.loss, 0.5)
        self.assertAlmostEqual(metrics.accuracy, 0.75)
        # One false positive (second 0 predicted as 1), no false negatives:
        # precision = 2/3, recall = 2/2 = 1.0, f1 = 2*P*R/(P+R) = 0.8.
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.f1, 0.8)
        # scores=[0,1,1,1] against labels=[0,0,1,1]: of the 4 positive/negative
        # score pairs, 2 rank correctly and 2 are tied (0.5 credit each):
        # (1+0.5+1+0.5)/4 = 0.75.
        self.assertAlmostEqual(metrics.auc, 0.75)
        # Per-class recall: class 0 is 1/2, class 1 is 2/2 -> balanced = 0.75.
        self.assertAlmostEqual(metrics.balanced_accuracy, 0.75)
        # po (observed agreement) = 3/4 = 0.75; pe (chance agreement) from
        # marginals (true 2/2, predicted 1/3) = (2*1 + 2*3)/16 = 0.5;
        # kappa = (po - pe) / (1 - pe) = 0.25 / 0.5 = 0.5.
        self.assertAlmostEqual(metrics.cohens_kappa, 0.5)

    def test_training_updates_parameters_and_evaluation_does_not(self) -> None:
        model = EEGGCN1(EEGGCN1Config(dropout=0.0))
        loader = _graphs()
        # The model outputs log-softmax, so NLLLoss is the correct pairing.
        loss_function = nn.NLLLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        training = train_epoch(
            model, loader, loss_function, optimizer, torch.device("cpu")
        )
        after = [parameter.detach().clone() for parameter in model.parameters()]
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))
        frozen = [parameter.detach().clone() for parameter in model.parameters()]
        evaluation = evaluate(model, loader, loss_function, torch.device("cpu"))
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(frozen, model.parameters()))
        )
        self.assertEqual(training.examples, 8)
        self.assertEqual(evaluation.examples, 8)

    def test_configuration_rejects_invalid_options(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=0)
        with self.assertRaises(ValueError):
            TrainingConfig(seed_strategy="random")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TrainingConfig(run_name="nested/run")
        with self.assertRaisesRegex(ValueError, "Unknown combination"):
            TrainingConfig(combination="not_a_real_combination")
        parser = build_parser()
        option_destinations = {
            action.dest for action in parser._actions  # type: ignore[attr-defined]
        }
        self.assertIn("combination", option_destinations)

    def test_default_combination_is_without_csd_alpha_wpli(self) -> None:
        self.assertEqual(TrainingConfig().combination, "without_csd_alpha_wpli")

    def test_fold_training_can_select_the_csd_combination(self) -> None:
        result = train_loso_fold(
            1,
            TrainingConfig(
                combination="csd_alpha_wpli",
                epochs=1,
                batch_size=64,
                seed=42,
                save_outputs=False,
            ),
            show_progress=False,
        )
        self.assertEqual(result.test_subject_id, 1)
        self.assertEqual(result.test.examples, 40)

    def test_public_fold_training_uses_six_features(self) -> None:
        result = train_loso_fold(
            1,
            TrainingConfig(
                epochs=1,
                batch_size=64,
                seed=42,
                seed_strategy="per_fold",
                save_outputs=False,
            ),
            show_progress=False,
        )
        self.assertEqual(result.test_subject_id, 1)
        self.assertEqual(result.initialization_seed, 43)
        self.assertEqual(result.test.examples, 40)
        self.assertEqual(result.epochs_ran, 1)

    def test_saved_fold_has_manifest_normalization_and_overwrite_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = TrainingConfig(
                run_name="test_run",
                output_dir=Path(temporary),
                epochs=1,
                batch_size=64,
                save_outputs=True,
            )
            train_loso_fold(1, config, show_progress=False)
            run_dir = Path(temporary) / "test_run"
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            checkpoint = torch.load(
                run_dir / "subject_01" / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["dataset_variant"], "without_csd_alpha_wpli")
            self.assertEqual(manifest["combination"]["node_variant"], "without_csd")
            self.assertEqual(manifest["combination"]["edge_variant"], "wpli_without_csd")
            self.assertEqual(manifest["combination"]["band_name"], "alpha")
            self.assertEqual(checkpoint["combination"], "without_csd_alpha_wpli")
            self.assertEqual(checkpoint["model_config"]["input_features"], 6)
            normalization = checkpoint["feature_normalization"]
            self.assertEqual(len(normalization["mean_by_subject"]), 50)
            for subject_mean in normalization["mean_by_subject"].values():
                self.assertEqual(tuple(subject_mean.shape), (6,))
            for subject_std in normalization["standard_deviation_by_subject"].values():
                self.assertEqual(tuple(subject_std.shape), (6,))
            with self.assertRaises(FileExistsError):
                train_loso_fold(1, config, show_progress=False)


def _synthetic_fold_results(values: list[float]) -> list[FoldResult]:
    """Build minimal FoldResults whose test accuracy is each given value."""

    return [
        FoldResult(
            test_subject_id=index + 1,
            validation_subject_ids=(),
            initialization_seed=42,
            best_epoch=1,
            epochs_ran=1,
            test=ClassificationMetrics(
                loss=1.0 - value,
                accuracy=value,
                balanced_accuracy=value,
                f1=value,
                recall=value,
                precision=value,
                auc=value,
                cohens_kappa=2 * value - 1,
                examples=40,
            ),
            history=(),
        )
        for index, value in enumerate(values)
    ]


class SummarizeResultsTests(unittest.TestCase):
    def test_confidence_intervals_bracket_the_mean_and_are_deterministic(self) -> None:
        results = _synthetic_fold_results(
            [0.40, 0.45, 0.50, 0.50, 0.55, 0.60, 0.65, 0.45, 0.55, 0.50]
        )
        summary = summarize_results(results, seed=42)
        repeated = summarize_results(results, seed=42)

        for metric_name in (
            "loss",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "recall",
            "precision",
            "auc",
            "cohens_kappa",
        ):
            mean = summary[f"mean_{metric_name}"]
            low = summary[f"ci95_low_{metric_name}"]
            high = summary[f"ci95_high_{metric_name}"]
            self.assertLessEqual(low, mean)
            self.assertLessEqual(mean, high)
            # Deterministic given a fixed seed.
            self.assertEqual(low, repeated[f"ci95_low_{metric_name}"])
            self.assertEqual(high, repeated[f"ci95_high_{metric_name}"])

    def test_different_seeds_can_change_bootstrap_bounds(self) -> None:
        results = _synthetic_fold_results(
            [0.40, 0.45, 0.50, 0.50, 0.55, 0.60, 0.65, 0.45, 0.55, 0.50]
        )
        first = summarize_results(results, seed=1, bootstrap_resamples=200)
        second = summarize_results(results, seed=2, bootstrap_resamples=200)
        self.assertNotEqual(
            (first["ci95_low_accuracy"], first["ci95_high_accuracy"]),
            (second["ci95_low_accuracy"], second["ci95_high_accuracy"]),
        )

    def test_rejects_invalid_confidence_level_and_resample_count(self) -> None:
        results = _synthetic_fold_results([0.5, 0.6])
        with self.assertRaises(ValueError):
            summarize_results(results, confidence_level=0.0)
        with self.assertRaises(ValueError):
            summarize_results(results, confidence_level=1.0)
        with self.assertRaises(ValueError):
            summarize_results(results, bootstrap_resamples=0)


if __name__ == "__main__":
    unittest.main()
