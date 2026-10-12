"""Scoring of every epoch, the seed ensemble, and the two experiment runs."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.data import Data

from ..model import ModelConfig
from ..train import TrainConfig, run_loso, run_within, within_config
from ..train.engine import ensemble_predictions, score_epochs
from ..train.loso import shuffled_labels
from ..train.metrics import classification_metrics, summarize
from ..train.run_loso import main
from ..train.run_within import main as within_main
from ..train.within import fold_indices
from .fixture import CONFIG, write_recordings


def one_node_graphs(labels):
    return [Data(x=torch.tensor([[0., 1.]] if label else [[1., 0.]]), y=torch.tensor(label),
                 edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 2)),
                 signal=torch.zeros(1, 4)) for label in labels]


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.folder = tempfile.TemporaryDirectory()
        write_recordings(cls.folder.name)
        cls.config = TrainConfig(epochs=3, batch_size=8, ensemble=2, device="cpu", show_progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_metrics(self):
        metrics = classification_metrics([0, 0, 0, 1], [0, 0, 1, 1])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 5 / 6)
        self.assertAlmostEqual(metrics["macro_f1"], 11 / 15)
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [0, 1]])
        summary = summarize({"1": metrics, "2": classification_metrics([0, 1], [0, 1])})
        self.assertEqual(summary["accuracy"], 0.875)
        self.assertEqual(summary["confusion_matrix"], [[3, 1], [0, 2]])
        with self.assertRaises(ValueError):
            classification_metrics([], [])

    def test_centring_removes_a_subject_wide_bias_without_labels(self):
        logits = torch.tensor([[[6., 0.], [5., 1.], [6., 0.], [5., 1.]]])   # one model, biased toward class 0
        self.assertEqual(ensemble_predictions(logits, center_logits=False).tolist(), [0, 0, 0, 0])
        self.assertEqual(ensemble_predictions(logits, center_logits=True).tolist(), [0, 1, 0, 1])

    def test_ensemble_averages_probabilities(self):
        # Two models: probabilities of class 1 are 0.9 and 0.2 (mean 0.55), then 0.6 and 0.2 (mean 0.4).
        probabilities = torch.tensor([[[0.1, 0.9], [0.4, 0.6]], [[0.8, 0.2], [0.8, 0.2]]])
        self.assertEqual(ensemble_predictions(probabilities.log(), center_logits=False).tolist(), [1, 0])

    def test_final_and_highest_epoch_scores(self):
        right, wrong = [0., 1.], [1., 0.]          # every trial is class 1
        epochs = [[right, wrong, wrong, wrong], [right, right, right, wrong],
                  [right, right, wrong, right], [right, wrong, wrong, wrong]]
        scores = score_epochs(torch.tensor([epochs]), [1, 1, 1, 1], center_logits=False)
        self.assertEqual(scores["test_accuracy_per_epoch"], [0.25, 0.75, 0.75, 0.25])
        self.assertEqual(scores["final_epoch"]["accuracy"], 0.25)
        self.assertEqual((scores["highest_epoch"]["accuracy"], scores["highest_epoch"]["epoch"]), (0.75, 2))
        self.assertEqual(scores["highest_epoch"]["confusion_matrix"], [[0, 0], [1, 3]])
        self.assertNotIn("epoch", scores["final_epoch"])

    def test_shuffled_labels_keep_the_class_counts_and_the_originals(self):
        graphs = one_node_graphs([0, 1] * 20)
        shuffled = shuffled_labels(graphs, seed=3)
        self.assertEqual([int(g.y) for g in graphs], [0, 1] * 20)
        self.assertEqual(sum(int(g.y) for g in shuffled), 20)
        self.assertNotEqual([int(g.y) for g in shuffled], [0, 1] * 20)
        self.assertTrue(all(torch.equal(a.x, b.x) for a, b in zip(graphs, shuffled)))
        self.assertEqual([int(g.y) for g in shuffled], [int(g.y) for g in shuffled_labels(graphs, seed=3)])

    def test_leave_one_subject_out_run(self):
        result = run_loso(self.folder.name, CONFIG, ModelConfig(), self.config)
        self.assertEqual(set(result["subjects"]), {"1", "2", "3"})
        for fold in result["subjects"].values():
            self.assertEqual((fold["training_trials"], fold["testing_trials"]), (16, 8))
            self.assertEqual(len(fold["test_accuracy_per_epoch"]), 3)
            self.assertEqual(fold["final_epoch"]["accuracy"], fold["test_accuracy_per_epoch"][-1])
            self.assertEqual(fold["highest_epoch"]["accuracy"], max(fold["test_accuracy_per_epoch"]))
            for score in ("final_epoch", "highest_epoch"):
                self.assertEqual(sum(map(sum, fold[score]["confusion_matrix"])), 8)
        for score in ("final_epoch", "highest_epoch"):
            self.assertEqual(sum(map(sum, result["summary"][score]["confusion_matrix"])), 24)
        self.assertGreaterEqual(result["summary"]["highest_epoch"]["accuracy"],
                                result["summary"]["final_epoch"]["accuracy"])
        self.assertFalse(result["shuffled_training_labels"])
        self.assertEqual(result["train_config"]["ensemble"], 2)
        self.assertTrue(result["unlabeled_test_subject_use"]["center_logits"])
        # The same seed gives the same result; subsets of held-out subjects can be run.
        again = run_loso(self.folder.name, CONFIG, ModelConfig(), self.config, held_out_subjects=(2,))
        self.assertEqual(again["subjects"]["2"], result["subjects"]["2"])
        with self.assertRaises(ValueError):
            run_loso(self.folder.name, CONFIG, ModelConfig(), self.config, held_out_subjects=(9,))
        control = run_loso(self.folder.name, CONFIG, ModelConfig(),
                           replace(self.config, shuffle_training_labels=True), held_out_subjects=(2,))
        self.assertTrue(control["shuffled_training_labels"])
        for bad in (dict(epochs=0), dict(ensemble=0), dict(learning_rate=0.0), dict(device="gpu")):
            with self.assertRaises(ValueError):
                replace(self.config, **bad).validate()

    def test_within_subject_folds_are_stratified_and_cover_every_trial_once(self):
        labels = [0, 1] * 10
        folds = fold_indices(labels, 5, 42)
        self.assertEqual(sorted(int(i) for _, test in folds for i in test), list(range(20)))
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            self.assertEqual(sum(labels[i] for i in test), 2)
        self.assertTrue(all((a[1] == b[1]).all() for a, b in zip(folds, fold_indices(labels, 5, 42))))
        with self.assertRaises(ValueError):
            fold_indices(labels, 1, 42)

    def test_within_subject_run(self):
        defaults = within_config()
        self.assertEqual((defaults.epochs, defaults.center_logits, defaults.ensemble), (60, False, 3))
        config = within_config(epochs=1, batch_size=8, ensemble=1, device="cpu", show_progress=False)
        result = run_within(self.folder.name, CONFIG, ModelConfig(), config, subjects=(1, 3), folds=2)
        self.assertEqual(set(result["subjects"]), {"1", "3"})
        for subject in result["subjects"].values():
            self.assertEqual(len(subject["folds"]), 2)
            for score in ("final_epoch", "highest_epoch"):                        # every trial tested once
                self.assertEqual(sum(map(sum, subject[score]["confusion_matrix"])), 8)
            self.assertEqual([f["testing_trials"] for f in subject["folds"]], [4, 4])
            self.assertEqual([f["training_trials"] for f in subject["folds"]], [4, 4])
            correct = subject["final_epoch"]["confusion_matrix"][0][0] + subject["final_epoch"]["confusion_matrix"][1][1]
            self.assertAlmostEqual(subject["final_epoch"]["accuracy"], correct / 8)
            self.assertGreaterEqual(subject["highest_epoch"]["accuracy"], subject["final_epoch"]["accuracy"])
        self.assertFalse(result["train_config"]["center_logits"])
        self.assertEqual(result["protocol"], "within_subject_cross_validation")
        with self.assertRaises(ValueError):
            run_within(self.folder.name, CONFIG, ModelConfig(), config, subjects=(9,))

    def test_within_command_line_reports_missing_recordings(self):
        with tempfile.TemporaryDirectory() as folder:
            write_recordings(folder, subjects=(1,))
            output = Path(folder) / "within.json"
            with self.assertRaises(SystemExit):   # the default subjects are not all present
                within_main(["--data-dir", folder, "--output", str(output), "--device", "cpu", "--quiet"])
            self.assertFalse(output.exists())

    def test_command_line_reports_missing_recordings(self):
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(SystemExit):
            main(["--data-dir", empty, "--output", str(Path(empty) / "result.json"), "--device", "cpu", "--quiet"])


if __name__ == "__main__":
    unittest.main()
