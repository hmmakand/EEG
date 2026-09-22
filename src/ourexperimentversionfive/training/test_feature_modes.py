"""Small end-to-end checks for both combinations and normalization modes."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from ..data.combinations import COMBINATIONS
from ..data.validation import load_dataset
from .config import TrainingConfig, WithinSubjectConfig
from .hyperparameter_search import select_hyperparameters
from .loso import train_loso_fold
from .within_subject import train_within_subject


class TrainingFeatureModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_modes_through_loso_within_subject_search_and_artifacts(self):
        for combination in COMBINATIONS.values():
            for mode in ("none", "zscore"):
                with self.subTest(combination=combination.name, mode=mode), tempfile.TemporaryDirectory() as temporary:
                    config = TrainingConfig(combination=combination.name, node_normalization=mode,
                        epochs=1, batch_size=128, device="cpu", run_name="smoke", output_dir=Path(temporary))
                    result = train_loso_fold(1, config, dataset=self.dataset, show_progress=False)
                    self.assertEqual(result.test.examples, 40)
                    run = Path(temporary) / "smoke"
                    manifest = json.loads((run / "run_manifest.json").read_text())
                    checkpoint = torch.load(run / "subject_01/best_model.pt", weights_only=False)
                    self.assertEqual(manifest["experiment"], "ourexperimentversionfive")
                    self.assertEqual(manifest["node_normalization"], mode)
                    self.assertEqual(manifest["combination"]["edge_variant"], combination.edge_variant)
                    self.assertEqual(checkpoint["model_config"]["input_features"], 8)
                    self.assertEqual(checkpoint["feature_selection"]["selected_node_indices"], list(range(8)))
                    self.assertEqual(checkpoint["feature_selection"]["schema_version"], 2)
                    self.assertEqual(Path(checkpoint["dataset_path"]), self.dataset.path)
                    norm = checkpoint["feature_normalization"]
                    self.assertEqual(norm["mode"], mode)
                    if mode == "none":
                        self.assertEqual(norm["mean_by_subject"], {})
                        self.assertEqual(norm["fit_graph_indices"], [])
                    else:
                        self.assertEqual(norm["fit_graph_indices"], checkpoint["split"]["train_graph_indices"])
                        torch.testing.assert_close(norm["mean_by_subject"][1], norm["mean_by_subject"][2])
                        self.assertEqual(tuple(norm["mean_by_subject"][1].shape), (8,))
                    within = train_within_subject(1, WithinSubjectConfig(combination=combination.name,
                        node_normalization=mode, folds=2, epochs=1, device="cpu", save_outputs=False),
                        dataset=self.dataset, combination=combination, show_progress=False)
                    self.assertEqual(len(within), 2)
                    for fold in within:
                        self.assertEqual(fold.preprocessing["mode"], mode)
                        if mode == "zscore":
                            self.assertEqual(fold.preprocessing["fit_graph_indices"], fold.preprocessing["train_graph_indices"])
                            self.assertFalse(set(fold.preprocessing["fit_graph_indices"]) & set(fold.preprocessing["evaluation_graph_indices"]))
                        else:
                            self.assertIsNone(fold.preprocessing["mean"])
                    best, scores = select_hyperparameters(combination, self.dataset, (1, 2, 3, 4),
                        grid=[{"learning_rate": 0.001, "weight_decay": 0.0005}], inner_folds=2,
                        search_epochs=1, base_config=config)
                    self.assertEqual(best["learning_rate"], 0.001)
                    self.assertEqual(len(scores[0]["per_fold_balanced_accuracy"]), 2)

    def test_search_cannot_change_normalization_mode(self):
        with self.assertRaisesRegex(ValueError, "fixed per run"):
            select_hyperparameters(COMBINATIONS["without_csd_alpha_wpli"], self.dataset, (1, 2, 3, 4),
                grid=[{"node_normalization": "zscore"}], inner_folds=2, search_epochs=1,
                base_config=TrainingConfig())
