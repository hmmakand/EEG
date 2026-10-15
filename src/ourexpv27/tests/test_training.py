"""Scoring, the seed ensemble, target calibration, and the leave-one-subject-out run."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

import json

import numpy as np

from ..train import TrainConfig, run_loso, run_within, within_config
from ..model import EEGGraphModel, ModelConfig
from ..train.calibration import calibrated_targets, class_templates
from ..train.engine import ensemble_predictions
from ..train.metrics import classification_metrics, summarize
from ..train.output import models_folder, result_path
from ..analysis.figures import main as figures
from ..train.run_loso import main, variant_name
from ..train.run_within import main as within_main
from ..train.within import fold_indices
from .fixture import BCI2A_CONFIG, CONFIG, fake_recordings, write_bci2a_recordings


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = TrainConfig(epochs=4, batch_size=8, ensemble=2, calibration_warmup_epochs=2,
                                 calibration_confidence=0.5, device="cpu", show_progress=False)

    def test_metrics(self):
        metrics = classification_metrics([0, 0, 0, 1], [0, 0, 1, 1])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 5 / 6)
        self.assertAlmostEqual(metrics["macro_f1"], 11 / 15)
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [0, 1]])
        self.assertAlmostEqual(metrics["kappa"], 0.5)
        self.assertAlmostEqual(metrics["precision"], (1.0 + 0.5) / 2)
        self.assertAlmostEqual(metrics["recall"], 5 / 6)
        self.assertIsNone(metrics["auc"])                                     # no scores given
        self.assertEqual(classification_metrics([0, 0, 1, 1], [0, 0, 0, 1], scores=[0.1, 0.4, 0.35, 0.8])["auc"], 0.75)
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

    def test_calibration_targets_need_confidence_and_motor_area_agreement(self):
        features = torch.tensor([[1., 0.], [0.9, 0.1], [-1., 0.], [-0.9, -0.1]])
        templates, thresholds = class_templates(features, torch.tensor([0, 0, 1, 1]), floor=0.0)
        torch.testing.assert_close(templates[0], -templates[1], atol=1e-6, rtol=0)
        self.assertTrue((thresholds > 0.9).all())
        logits = torch.tensor([[4., 0.],     # confident class 0, motor areas agree       -> target 0
                               [4., 0.],     # confident class 0, motor areas say class 1 -> no target
                               [0.2, 0.],    # not confident                              -> no target
                               [0., 4.]])    # confident class 1, motor areas agree       -> target 1
        sample_features = torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]])
        targets = calibrated_targets(logits, sample_features, templates, thresholds, confidence=0.9,
                                     center_logits=False, existing=torch.zeros(4, 2))
        self.assertEqual(targets.tolist(), [[1., 0.], [0., 0.], [0., 0.], [0., 1.]])
        # A sample keeps its target until a prediction that passes both checks replaces it.
        later = calibrated_targets(logits.flip(1), sample_features, templates, thresholds, 0.9, False, targets)
        self.assertEqual(later.tolist(), [[1., 0.], [0., 1.], [0., 0.], [0., 1.]])
        # Centring removes a bias toward one class before the confidence is judged.
        biased = torch.tensor([[9., 0.], [5., 4.]])
        centred = calibrated_targets(biased, torch.tensor([[1., 0.], [-1., 0.]]), templates, thresholds, 0.85, True,
                                     torch.zeros(2, 2))
        self.assertEqual(centred.tolist(), [[1., 0.], [0., 1.]])

    def test_leave_one_subject_out_run(self):
        with fake_recordings():
            result = run_loso("folder", CONFIG, config=self.config)
            again = run_loso("folder", CONFIG, config=self.config, held_out_subjects=(2,))
            with self.assertRaises(ValueError):
                run_loso("folder", CONFIG, config=self.config, held_out_subjects=(9,))
        self.assertEqual(set(result["subjects"]), {"1", "2", "3"})
        for fold in result["subjects"].values():
            self.assertEqual((fold["training_samples"], fold["testing_samples"]), (16, 8))
            self.assertEqual(sum(map(sum, fold["confusion_matrix"])), 8)
            self.assertEqual(len(fold["calibration"]), 2)                       # one entry per model of the ensemble
            self.assertTrue(all(0.0 <= entry["labelled_fraction"] <= 1.0 for entry in fold["calibration"]))
            # The result is the final epoch; the highest test epoch is recorded next to it.
            self.assertEqual(len(fold["test_accuracy_per_epoch"]), 4)
            self.assertEqual(fold["accuracy"], fold["test_accuracy_per_epoch"][-1])
            self.assertEqual(fold["highest_epoch"]["accuracy"], max(fold["test_accuracy_per_epoch"]))
            self.assertTrue(0.0 <= fold["auc"] <= 1.0)
            self.assertEqual(len(fold["scores"]), 8)
        self.assertEqual(sum(map(sum, result["summary"]["confusion_matrix"])), 24)
        self.assertGreaterEqual(result["summary"]["highest_epoch"]["accuracy"], result["summary"]["accuracy"])
        self.assertEqual(set(result["summary"]) - {"confusion_matrix", "highest_epoch"},
                         {"accuracy", "balanced_accuracy", "macro_f1", "kappa", "precision", "recall", "auc"})
        self.assertEqual((result["protocol"], result["dataset"], result["class_names"]),
                         ("leave_one_subject_out", "liu2024", ["rest", "imagery"]))
        self.assertEqual((result["model_config"]["num_electrodes"], result["model_config"]["signal_samples"]), (29, 475))
        self.assertTrue(result["unlabeled_test_subject_use"]["center_logits"])
        # Training curves, cost and model size are recorded.
        fold = result["subjects"]["2"]
        self.assertEqual(len(fold["history"]), 2)                               # one per model
        for history in fold["history"]:
            for key in ("train_loss", "train_accuracy", "test_loss", "test_accuracy",
                        "calibration_labelled_fraction", "calibration_target_accuracy"):
                self.assertEqual(len(history[key]), 4, key)
            self.assertGreater(history["training_seconds"], 0)
        self.assertEqual(len(fold["test_loss_per_epoch"]), 4)
        self.assertGreater(result["cost"]["training_seconds_total"], 0)
        size = result["complexity"]
        self.assertEqual(size["parameters"], sum(p.numel() for p in EEGGraphModel(ModelConfig()).parameters()))
        self.assertEqual(sum(size["parameters_per_path"].values()), size["parameters"])
        self.assertGreater(size["flops_per_sample"], 0)
        self.assertGreater(size["latency_ms_per_sample"], 0)
        self.assertGreater(size["model_file_bytes"], 4 * size["parameters"])
        # Embeddings of every test sample: graph head input and temporal features.
        self.assertEqual(result["embeddings"]["2"]["graph"].shape, (8, 64))
        self.assertEqual(result["embeddings"]["2"]["temporal"].shape[0], 8)
        # The same seed gives the same result (apart from the measured times); subsets of held-out subjects can be run.
        timeless = lambda fold: {key: value for key, value in fold.items() if key not in ("training_seconds", "history")}
        self.assertEqual(timeless(again["subjects"]["2"]), timeless(result["subjects"]["2"]))
        self.assertEqual([{k: v for k, v in h.items() if k != "training_seconds"} for h in again["subjects"]["2"]["history"]],
                         [{k: v for k, v in h.items() if k != "training_seconds"} for h in fold["history"]])

    def test_ablations(self):
        # Every path alone, and without calibration; the method is the sum of all three.
        with fake_recordings():
            results = {paths: run_loso("folder", CONFIG, self.config, held_out_subjects=(2,), paths=paths)
                       for paths in (("graph",), ("wide",), ("temporal",), ("graph", "temporal"))}
            plain = run_loso("folder", CONFIG, replace(self.config, target_calibration=False), held_out_subjects=(2,))
        for paths, result in results.items():
            self.assertEqual(tuple(result["model_config"]["paths"]), paths)
            self.assertEqual(set(result["embeddings"]["2"]), {"graph", "temporal"} & set(paths))
        self.assertNotIn("calibration", plain["subjects"]["2"])
        self.assertNotIn("calibration_labelled_fraction", plain["subjects"]["2"]["history"][0])
        with self.assertRaises(ValueError):
            ModelConfig(paths=("graph", "other"))
        # With all paths the selected-path computation equals the full one.
        torch.manual_seed(0)
        model = EEGGraphModel(ModelConfig()).eval()
        from ..datascripts import build_subject_graphs
        from torch_geometric.loader import DataLoader
        with fake_recordings():
            data = next(iter(DataLoader(build_subject_graphs("folder", 1, CONFIG), batch_size=4)))
        full = model(data.x, data.edge_index, data.batch, data.edge_attr, data.signal)
        model.config = replace(model.config, paths=("graph", "wide"))
        two = model(data.x, data.edge_index, data.batch, data.edge_attr, data.signal)
        model.config = replace(model.config, paths=("temporal",))
        one = model(data.x, data.edge_index, data.batch, data.edge_attr, data.signal)
        torch.testing.assert_close(two + one, full)
        self.assertEqual(variant_name(("graph", "wide", "temporal")), "")
        self.assertEqual(variant_name(("wide", "graph"), False), "Paths-graph-wide_NoCalibration")

    def test_within_subject_folds_keep_trials_together(self):
        # Two samples per trial (liu2024): both go to the same fold; every sample is tested once.
        trial_ids = [t for t in range(10) for _ in (0, 1)]
        labels = [1, 0] * 10
        folds = fold_indices(trial_ids, labels, 5, 42)
        self.assertEqual(sorted(int(i) for _, test in folds for i in test), list(range(20)))
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            self.assertEqual(sorted(trial_ids[i] for i in test), sorted(2 * [trial_ids[i] for i in test[::2]]))
            self.assertEqual(sum(labels[i] for i in test), 2)
        # One sample per trial (bci2a): the folds are stratified by class.
        labels = [0, 1] * 10
        for _, test in fold_indices(range(20), labels, 5, 42):
            self.assertEqual(sum(labels[i] for i in test), 2)
        self.assertTrue(all(np.array_equal(a[1], b[1]) for a, b in zip(folds, fold_indices(trial_ids, [1, 0] * 10, 5, 42))))
        with self.assertRaises(ValueError):
            fold_indices(trial_ids, labels, 1, 42)

    def test_within_subject_run(self):
        defaults = within_config()
        self.assertEqual((defaults.epochs, defaults.center_logits, defaults.target_calibration, defaults.ensemble),
                         (60, False, False, 3))
        config = within_config(epochs=1, batch_size=8, ensemble=1, device="cpu", show_progress=False)
        with fake_recordings():
            result = run_within("folder", CONFIG, config, subjects=(1, 3), folds=2)
            with self.assertRaises(ValueError):
                run_within("folder", CONFIG, config, subjects=(9,))
            with self.assertRaises(ValueError):                                # no calibration within a subject
                run_within("folder", CONFIG, within_config(target_calibration=True, device="cpu"))
        self.assertEqual(set(result["subjects"]), {"1", "3"})
        for subject in result["subjects"].values():
            self.assertEqual([(f["training_samples"], f["testing_samples"]) for f in subject["folds"]], [(4, 4)] * 2)
            self.assertEqual(sum(map(sum, subject["confusion_matrix"])), 8)       # every sample tested once
            self.assertNotIn("calibration", subject["folds"][0])
            self.assertGreaterEqual(subject["highest_epoch"]["accuracy"], subject["accuracy"])
        self.assertEqual(result["protocol"], "within_subject_cross_validation")

    def test_bci2a_runs_and_outputs_go_into_the_folder_of_the_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            write_bci2a_recordings(folder)
            result = run_loso(folder, BCI2A_CONFIG, self.config, held_out_subjects=(1,))
            self.assertEqual((result["dataset"], result["class_names"]), ("bci2a", ["left hand", "right hand"]))
            self.assertEqual((result["model_config"]["num_electrodes"], result["model_config"]["signal_samples"]), (22, 500))
            self.assertEqual(result["subjects"]["1"]["training_samples"], 16)
            # Result files: <output-dir>/<dataset>/..., with a time stamp when the name is taken.
            out = Path(folder) / "outputs"
            first = result_path(out, "bci2a", "LOSO", 12345)
            self.assertEqual(first, out / "bci2a" / "Version27_bci2a_LeftRight_LOSO_seed12345.json")
            self.assertEqual(result_path(out, "liu2024", "Within", 7),
                             out / "liu2024" / "Version27_liu2024_ImageryRest_Within_seed7.json")
            first.parent.mkdir(parents=True)
            first.write_text("{}")
            second = result_path(out, "bci2a", "LOSO", 12345)
            self.assertNotEqual(second, first)
            self.assertEqual(second.parent, first.parent)
            self.assertEqual(models_folder(first), out / "bci2a" / "models" / first.stem)

    def test_models_can_be_saved_and_loaded(self):
        with tempfile.TemporaryDirectory() as folder, fake_recordings():
            models = Path(folder) / "models"
            run_loso("folder", CONFIG, self.config, held_out_subjects=(2,), models_dir=models)
            self.assertEqual(sorted(p.name for p in models.iterdir()), ["S2_model1.pt", "S2_model2.pt"])
            saved = torch.load(models / "S2_model1.pt", weights_only=True)
            self.assertEqual((saved["seed"], saved["data_config"]["dataset"]), (12345, "liu2024"))
            model = EEGGraphModel(ModelConfig(**saved["model_config"]))
            model.load_state_dict(saved["state_dict"])                         # the settings rebuild the model

    def test_settings_are_validated(self):
        defaults = TrainConfig()
        self.assertEqual((defaults.epochs, defaults.ensemble, defaults.calibration_warmup_epochs,
                          defaults.target_calibration), (20, 3, 10, True))
        for bad in (dict(epochs=0), dict(ensemble=0), dict(learning_rate=0.0), dict(device="gpu"),
                    dict(calibration_warmup_epochs=21), dict(calibration_confidence=0.2),
                    dict(calibration_source_weight=0.0)):
            with self.assertRaises(ValueError):
                replace(defaults, **bad).validate()

    def test_command_line_reports_missing_recordings(self):
        for command, dataset in ((main, "liu2024"), (within_main, "bci2a")):
            with tempfile.TemporaryDirectory() as empty, self.assertRaises(SystemExit):
                command(["--dataset", dataset, "--data-dir", empty, "--output-dir", empty, "--device", "cpu", "--quiet"])

    def test_command_line_writes_into_the_folder_of_the_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            write_bci2a_recordings(folder, subjects=(1, 2, 3, 5, 6, 7, 8, 9))
            out = Path(folder) / "out"
            within_main(["--dataset", "bci2a", "--data-dir", folder, "--output-dir", str(out), "--subjects", "1",
                         "--folds", "2", "--epochs", "1", "--device", "cpu", "--quiet", "--save-models"])
            result_file = out / "bci2a" / "Version27_bci2a_LeftRight_Within_seed12345.json"
            self.assertEqual(json.loads(result_file.read_text())["protocol"], "within_subject_cross_validation")
            saved = json.loads(result_file.read_text())
            self.assertIn("highest_epoch", saved["summary"])                   # logged with the result, not only printed
            self.assertIn("highest_epoch", saved["subjects"]["1"])
            table = (out / "bci2a" / "Version27_bci2a_LeftRight_Within_seed12345.summary.txt").read_text()
            self.assertIn("highest test epoch", table)
            self.assertIn("kappa", table)
            saved = sorted(p.name for p in (out / "bci2a" / "models" / result_file.stem).iterdir())
            self.assertEqual(len(saved), 2 * 3)                                # two folds, three models each
            self.assertEqual(saved[0], "S1_fold1_model1.pt")
            self.assertIn("parameters", table)
            features = np.load(out / "bci2a" / "Version27_bci2a_LeftRight_Within_seed12345.embeddings.npz")
            self.assertEqual(features["graph_S1"].shape[0], len(json.loads(result_file.read_text())["subjects"]["1"]
                                                                ["folds"][0]["labels"]) * 2)
            main(["--dataset", "bci2a", "--data-dir", folder, "--output-dir", str(out), "--subjects", "1",
                  "--device", "cpu", "--quiet", "--paths", "temporal", "--no-target-calibration"])
            ablation = out / "bci2a" / "Version27_bci2a_LeftRight_LOSO_Paths-temporal_NoCalibration_seed12345.json"
            self.assertIn("Ablation", ablation.with_name(ablation.stem + ".summary.txt").read_text())
            figures([str(result_file), str(ablation)])                        # ROC, curves, t-SNE, model table
            drawn = sorted(p.name for p in (out / "bci2a" / (result_file.stem + "_figures")).iterdir())
            self.assertEqual(drawn, ["confusion_matrix.png", "model.txt", "roc.png", "roc_comparison.png",
                                     "subjects.png", "training_curves.png", "tsne.png"])


if __name__ == "__main__":
    unittest.main()
