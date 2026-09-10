"""Tests for the dedicated broadcast-11 training workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ourexperimentversionthree.model import EEGGCN1, EEGGCN1Config
from src.ourexperimentversionthree.training import (
    TrainingConfig,
    evaluate,
    train_epoch,
    train_loso_fold,
)
from src.ourexperimentversionthree.training.metrics import calculate_metrics
from src.ourexperimentversionthree.training.train import build_parser


def _graphs() -> TensorDataset:
    """Build 8 deterministic dense (x, adj, y) graphs for engine-level tests."""

    generator = torch.Generator().manual_seed(42)
    x = torch.stack(
        [
            torch.randn(29, 11, generator=generator) + graph_index % 2
            for graph_index in range(8)
        ]
    )
    ring = torch.eye(29, dtype=torch.float32)
    ring += torch.roll(torch.eye(29, dtype=torch.float32), shifts=1, dims=0)
    ring += torch.roll(torch.eye(29, dtype=torch.float32), shifts=-1, dims=0)
    adjacency = (ring > 0).to(torch.float32).unsqueeze(0).repeat(8, 1, 1)
    labels = torch.tensor([graph_index % 2 for graph_index in range(8)], dtype=torch.long)
    return TensorDataset(x, adjacency, labels)


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

    def test_training_updates_parameters_and_evaluation_does_not(self) -> None:
        model = EEGGCN1(EEGGCN1Config(dropout=0.0))
        loader = DataLoader(_graphs(), batch_size=4, shuffle=False)
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

    def test_configuration_and_cli_are_fixed_to_one_dataset(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=0)
        with self.assertRaises(ValueError):
            TrainingConfig(seed_strategy="random")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TrainingConfig(run_name="nested/run")
        parser = build_parser()
        option_destinations = {
            action.dest for action in parser._actions  # type: ignore[attr-defined]
        }
        self.assertNotIn("dataset_version", option_destinations)

    def test_public_fold_training_uses_eleven_features(self) -> None:
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
            self.assertEqual(manifest["dataset_variant"], "manuscript_broadcast_11_v3")
            self.assertEqual(checkpoint["model_config"]["input_features"], 11)
            self.assertEqual(
                tuple(checkpoint["feature_normalization"]["mean"].shape),
                (11,),
            )
            with self.assertRaises(FileExistsError):
                train_loso_fold(1, config, show_progress=False)


if __name__ == "__main__":
    unittest.main()
