"""Evaluation, the seed ensemble, and the leave-one-subject-out run."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..train import TrainConfig, run_loso, run_within, within_config
from ..train.engine import Ensemble, ensemble_predictions, evaluate, score_epochs
from ..train.calibration import calibrated_targets, class_templates
from ..train.loso import shuffled_labels
from ..train.metrics import classification_metrics, summarize
from ..train.run_loso import main
from ..train.run_within import main as within_main
from ..train.within import fold_indices
from .fixture import CONFIG, write_recordings


class Fixed(torch.nn.Module):
    """Returns x as logits, plus a constant bias toward class 0."""

    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = torch.tensor([bias, 0.0])

    def forward(self, x, edge_index, batch, edge_attr, signal, signal_low=None):
        return x + self.bias


class Constant(torch.nn.Module):
    """Returns fixed logits, whatever the input."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, x, edge_index, batch, edge_attr, signal, signal_low=None):
        return self.logits


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
        cls.config = TrainConfig(epochs=1, batch_size=8, ensemble=2, device="cpu", show_progress=False)

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
        loader = DataLoader(one_node_graphs([0, 1, 0, 1]), batch_size=3)
        labels, biased = evaluate(Fixed(bias=5.0), loader, torch.device("cpu"), center_logits=False)
        self.assertEqual((labels, biased), ([0, 1, 0, 1], [0, 0, 0, 0]))
        _, centred = evaluate(Fixed(bias=5.0), loader, torch.device("cpu"), center_logits=True)
        self.assertEqual(centred, [0, 1, 0, 1])

    def test_ensemble_averages_probabilities(self):
        batch = next(iter(DataLoader(one_node_graphs([0, 1]), batch_size=2)))
        members = [Fixed(0.0), Fixed(2.0), Fixed(-1.0)]
        inputs = (batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.signal)
        expected = torch.stack([m(*inputs).softmax(-1) for m in members]).mean(0)
        torch.testing.assert_close(Ensemble(members)(*inputs).exp(), expected)

    def test_leave_one_subject_out_run(self):
        result = run_loso(self.folder.name, CONFIG, config=self.config)
        self.assertEqual(result["dataset"], "bci2a")
        self.assertEqual((result["model_config"]["num_electrodes"], result["model_config"]["signal_samples"]), (22, 500))
        self.assertEqual(set(result["subjects"]), {"1", "2", "3"})
        for fold in result["subjects"].values():
            self.assertEqual((fold["training_trials"], fold["testing_trials"]), (16, 8))
            self.assertEqual(sum(map(sum, fold["confusion_matrix"])), 8)
        self.assertEqual(sum(map(sum, result["summary"]["confusion_matrix"])), 24)
        self.assertEqual(result["train_config"]["ensemble"], 2)
        self.assertTrue(result["unlabeled_test_subject_use"]["center_logits"])
        # The same seed gives the same result; subsets of held-out subjects can be run.
        again = run_loso(self.folder.name, CONFIG, config=self.config, held_out_subjects=(2,))
        self.assertEqual(again["subjects"]["2"], result["subjects"]["2"])
        with self.assertRaises(ValueError):
            run_loso(self.folder.name, CONFIG, config=self.config, held_out_subjects=(9,))
        for bad in (dict(epochs=0), dict(ensemble=0), dict(learning_rate=0.0), dict(device="gpu")):
            with self.assertRaises(ValueError):
                replace(self.config, **bad).validate()

    def test_final_and_highest_epoch_scores(self):
        right, wrong = [0., 1.], [1., 0.]          # every trial is class 1
        epochs = [[right, wrong, wrong, wrong], [right, right, right, wrong],
                  [right, right, wrong, right], [right, wrong, wrong, wrong]]
        scores = score_epochs(torch.tensor([epochs]), [1, 1, 1, 1], center_logits=False)
        self.assertEqual(scores["test_accuracy_per_epoch"], [0.25, 0.75, 0.75, 0.25])
        self.assertEqual(scores["accuracy"], 0.25)                                    # the result: final epoch
        self.assertEqual((scores["highest_epoch"]["accuracy"], scores["highest_epoch"]["epoch"]), (0.75, 2))
        # The per-epoch decision is that of evaluate() on an Ensemble.
        logits = torch.tensor([[[2., 0.], [0., 1.], [3., 0.]], [[0., 1.], [0., 3.], [2., 0.]]])
        loader = DataLoader(one_node_graphs([0, 1, 0]), batch_size=3)
        members = [Constant(logits[0]), Constant(logits[1])]
        for centre in (False, True):
            _, expected = evaluate(Ensemble(members), loader, torch.device("cpu"), center_logits=centre)
            self.assertEqual(ensemble_predictions(logits, centre).tolist(), expected)

    def test_run_records_the_highest_epoch_and_the_shuffled_label_control(self):
        config = replace(self.config, epochs=3)
        result = run_loso(self.folder.name, CONFIG, config=config, held_out_subjects=(1,))
        fold = result["subjects"]["1"]
        self.assertEqual(len(fold["test_accuracy_per_epoch"]), 3)
        self.assertEqual(fold["accuracy"], fold["test_accuracy_per_epoch"][-1])
        self.assertEqual(fold["highest_epoch"]["accuracy"], max(fold["test_accuracy_per_epoch"]))
        self.assertGreaterEqual(result["summary"]["highest_epoch"]["accuracy"], result["summary"]["accuracy"])
        self.assertFalse(result["shuffled_training_labels"])
        graphs = one_node_graphs([0, 1] * 20)
        shuffled = shuffled_labels(graphs, seed=3)
        self.assertEqual([int(g.y) for g in graphs], [0, 1] * 20)
        self.assertEqual(sum(int(g.y) for g in shuffled), 20)
        self.assertNotEqual([int(g.y) for g in shuffled], [0, 1] * 20)
        control = run_loso(self.folder.name, CONFIG, config=replace(config, shuffle_training_labels=True),
                           held_out_subjects=(1,))
        self.assertTrue(control["shuffled_training_labels"])

    def test_command_line_can_run_on_a_chosen_list_of_subjects(self):
        with tempfile.TemporaryDirectory() as folder:
            write_recordings(folder, subjects=(2, 4, 6))
            output = Path(folder) / "result.json"
            main(["--dataset", "bci2a", "--data-dir", folder, "--output", str(output), "--device", "cpu", "--quiet",
                  "--use-subjects", "2", "4", "6", "--subjects", "4", "--epochs", "2"])
            result = json.loads(output.read_text())
            self.assertEqual(result["data_config"]["subjects"], [2, 4, 6])       # 4 is not in the default list
            self.assertEqual(result["train_config"]["epochs"], 2)
            self.assertEqual(result["subjects"]["4"]["training_trials"], 24)
            with self.assertRaises(SystemExit):
                main(["--dataset", "bci2a", "--data-dir", folder, "--output", str(output), "--use-subjects", "2", "2"])

    def test_calibration_targets_need_confidence_and_motor_area_agreement(self):
        features = torch.tensor([[1., 0.], [0.9, 0.1], [-1., 0.], [-0.9, -0.1]])
        templates, thresholds = class_templates(features, torch.tensor([0, 0, 1, 1]), floor=0.0)
        torch.testing.assert_close(templates[0], -templates[1], atol=1e-6, rtol=0)
        self.assertTrue((thresholds > 0.9).all())
        logits = torch.tensor([[4., 0.],     # confident class 0, motor areas agree      -> target 0
                               [4., 0.],     # confident class 0, motor areas say class 1 -> no target
                               [0.2, 0.],    # not confident                              -> no target
                               [0., 4.]])    # confident class 1, motor areas agree      -> target 1
        trial_features = torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]])
        targets = calibrated_targets(logits, trial_features, templates, thresholds, confidence=0.9,
                                     center_logits=False, existing=torch.zeros(4, 2))
        self.assertEqual(targets.tolist(), [[1., 0.], [0., 0.], [0., 0.], [0., 1.]])
        # A trial keeps its target until a prediction that passes both checks replaces it.
        later = calibrated_targets(logits.flip(1), trial_features, templates, thresholds, 0.9, False, targets)
        self.assertEqual(later.tolist(), [[1., 0.], [0., 1.], [0., 0.], [0., 1.]])
        # Centring removes a bias toward one class before the confidence is judged.
        biased = torch.tensor([[9., 0.], [5., 4.]])
        centred = calibrated_targets(biased, torch.tensor([[1., 0.], [-1., 0.]]), templates, thresholds, 0.85, True,
                                     torch.zeros(2, 2))
        self.assertEqual(centred.tolist(), [[1., 0.], [0., 1.]])

    def test_leave_one_subject_out_with_target_calibration(self):
        config = replace(self.config, epochs=4, target_calibration=True, calibration_warmup_epochs=2,
                         calibration_confidence=0.5)
        result = run_loso(self.folder.name, CONFIG, config=config, held_out_subjects=(1,))
        fold = result["subjects"]["1"]
        self.assertEqual(len(fold["calibration"]), 2)                       # one entry per model of the ensemble
        self.assertTrue(all(0.0 <= entry["labelled_fraction"] <= 1.0 for entry in fold["calibration"]))
        self.assertTrue(result["train_config"]["target_calibration"])
        # The warm-up epochs are the ordinary training: same scores as without calibration.
        plain = run_loso(self.folder.name, CONFIG, config=replace(self.config, epochs=4, target_calibration=False),
                         held_out_subjects=(1,))
        self.assertEqual(fold["test_accuracy_per_epoch"][:2], plain["subjects"]["1"]["test_accuracy_per_epoch"][:2])
        self.assertNotIn("calibration", plain["subjects"]["1"])
        with self.assertRaises(ValueError):                                # within subject there is no unseen subject
            run_within(self.folder.name, CONFIG, config=within_config(target_calibration=True, device="cpu"))
        with self.assertRaises(ValueError):
            replace(config, calibration_warmup_epochs=5).validate()
        # Calibration is the default for leave one subject out, with half of the epochs as warm-up.
        defaults = TrainConfig()
        self.assertEqual((defaults.target_calibration, defaults.warmup_epochs), (True, 10))
        self.assertEqual(replace(defaults, epochs=7).warmup_epochs, 3)
        self.assertFalse(within_config().target_calibration)

    def test_runs_with_tangent_features_and_the_low_frequency_path(self):
        data_config = replace(CONFIG, tangent_features=True, low_band=(0.5, 4.0))
        result = run_loso(self.folder.name, data_config, config=replace(self.config, epochs=2), held_out_subjects=(1,))
        model = result["model_config"]
        self.assertEqual((model["node_features"], model["edge_features"], model["low_samples"]), (7, 3, 50))
        self.assertTrue(result["data_config"]["tangent_features"])
        self.assertEqual(sum(map(sum, result["subjects"]["1"]["confusion_matrix"])), 8)

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
        result = run_within(self.folder.name, CONFIG, config=config, subjects=(1, 3), folds=2)
        self.assertEqual(set(result["subjects"]), {"1", "3"})
        for subject in result["subjects"].values():
            self.assertEqual(len(subject["folds"]), 2)
            self.assertEqual(sum(map(sum, subject["confusion_matrix"])), 8)       # every trial tested once
            self.assertEqual([f["testing_trials"] for f in subject["folds"]], [4, 4])
            self.assertEqual([f["training_trials"] for f in subject["folds"]], [4, 4])
            correct = subject["confusion_matrix"][0][0] + subject["confusion_matrix"][1][1]
            self.assertAlmostEqual(subject["accuracy"], correct / 8)
        self.assertFalse(result["train_config"]["center_logits"])
        self.assertEqual(result["protocol"], "within_subject_cross_validation")
        with self.assertRaises(ValueError):
            run_within(self.folder.name, CONFIG, config=config, subjects=(9,))

    def test_within_command_line_reports_missing_recordings(self):
        with tempfile.TemporaryDirectory() as folder:
            write_recordings(folder, subjects=(1,))
            output = Path(folder) / "within.json"
            with self.assertRaises(SystemExit):   # the default subjects are not all present
                within_main(["--dataset", "bci2a", "--data-dir", folder, "--output", str(output), "--device", "cpu", "--quiet"])
            self.assertFalse(output.exists())

    def test_command_line_can_exclude_subjects_from_training_and_testing(self):
        with tempfile.TemporaryDirectory() as folder:
            write_recordings(folder, subjects=(1, 2, 3, 5, 6))
            output = Path(folder) / "result.json"
            main(["--dataset", "bci2a", "--data-dir", folder, "--output", str(output), "--device", "cpu", "--quiet",
                  "--exclude-subjects", "7", "8", "9", "--subjects", "1"])
            result = json.loads(output.read_text())
            self.assertEqual(result["data_config"]["subjects"], [1, 2, 3, 5, 6])
            self.assertEqual(result["subjects"]["1"]["training_trials"], 48)   # the four other subjects
            with self.assertRaises(SystemExit):   # subject 4 is not part of the default experiment
                main(["--dataset", "bci2a", "--data-dir", folder, "--output", str(output), "--exclude-subjects", "4"])

    def test_command_line_reports_missing_recordings(self):
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(SystemExit):
            main(["--data-dir", empty, "--output", str(Path(empty) / "result.json"), "--device", "cpu", "--quiet"])
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(SystemExit):   # the default dataset is liu2024
            main(["--data-dir", empty, "--all-subjects", "--device", "cpu", "--quiet"])


if __name__ == "__main__":
    unittest.main()
