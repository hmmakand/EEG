"""PA-TCNet: settings, data preparation and model."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ..datascripts import SAMPLE_RATE, Trial, liu2024
from ..evaluation.complexity import environment, model_complexity
from ..papers.patcnet.channels import ROI_GROUPS, channel_info
from ..papers.patcnet.network import build_model
from ..papers.patcnet.original.physiology import build_class_templates, compute_roi_features
from ..papers.patcnet.preparation import load_comparison_subject, load_subject, prepare_trials
from ..papers.patcnet.settings import DATASETS, XW_STROKE_SUBJECTS, TrainingSettings, dataset_settings, flattened_features
from ..papers.patcnet.training import fold_scores, loader, run_fold, stopping_epoch

LIU = Path(os.environ.get("LIU2024_DATA", "/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data"))
JIA = Path(os.environ.get("JIA2019_DATA", "/workspaces/datadisk/databci/eegdata/jia2019"))


class PATCNetTests(unittest.TestCase):
    def test_settings_follow_the_paper(self):
        liu, jia = DATASETS["liu2024"], DATASETS["jia2019"]
        self.assertEqual((len(XW_STROKE_SUBJECTS), liu.channels, liu.samples), (24, 30, 1000))
        self.assertEqual((len(jia.subjects), jia.channels, jia.samples), (15, 63, 1708))
        self.assertEqual((liu.source_weight, liu.confidence, liu.roi_floor, liu.warmup_epochs), (0.98, 0.60, 0.50, 25))
        self.assertEqual((jia.source_weight, jia.confidence, jia.roi_floor, jia.warmup_epochs), (0.95, 0.60, 0.45, 10))
        self.assertEqual((flattened_features(1000), flattened_features(1708)), (450, 780))
        self.assertTrue({47, 49} <= set(XW_STROKE_SUBJECTS))          # the duplicated pair is in the cohort

    def test_preparation(self):
        rng = np.random.default_rng(0)
        trials = [Trial(rng.normal(size=(2000, 30)) * 1e-5, 500, label, 1, i) for i, label in enumerate((0, 1, 0))]
        x, y = prepare_trials(trials, (0.0, 4.0))
        self.assertEqual((x.shape, x.dtype, y.tolist()), ((3, 30, 1000), np.float32, [0, 1, 0]))
        np.testing.assert_allclose(x.mean(axis=1), 0, atol=1e-4)          # common average reference
        np.testing.assert_allclose(x.mean(axis=2), 0, atol=1e-4)          # baseline
        self.assertTrue(1.0 < x.std() < 100.0)                            # microvolts
        whole, _ = prepare_trials(trials, None)
        self.assertEqual(whole.shape, (3, 30, 2000))
        with self.assertRaises(ValueError):
            prepare_trials(trials, (0.0, 8.0))

    def test_model_shapes(self):
        torch.manual_seed(0)
        for dataset, shape in (("liu2024", (30, 1000)), ("jia2019", (63, 1708)), ("bci2a", (22, 500))):
            model = build_model(dataset).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.randn(3, *shape)).shape), (3, 2))

    def test_roi_groups(self):
        info = channel_info("liu2024")
        self.assertEqual(len(info["channel_names"]), 30)
        self.assertTrue(all(name in info["channel_names"] for group in ROI_GROUPS.values() for name in group))
        with self.assertRaises(ValueError):
            channel_info("jia2019")
        x = np.random.default_rng(1).normal(size=(6, 30, 1000)).astype(np.float32)
        roi = compute_roi_features(x, info)["roi_vector"]
        self.assertEqual(roi.shape, (6, 4000))
        templates, thresholds = build_class_templates(roi, np.array([0, 1] * 3), 2, floor=0.5)
        self.assertEqual((templates.shape, thresholds.shape), ((2, 4000), (2,)))

    def test_evaluation_keeps_singleton_remainder(self):
        tensors = (torch.arange(9), torch.arange(9))
        self.assertEqual(sum(len(x) for x, _ in loader(tensors, 8, False)), 9)
        self.assertEqual(sum(len(x) for x, _ in loader(tensors, 8, True)), 8)

    def test_bci_preparation_and_roi(self):
        rng = np.random.default_rng(12)
        trials = [Trial(rng.normal(size=(1000, 22)) * 10, 500, y, 1, i)
                  for i, y in enumerate([0, 1] * 10)]
        with patch("src.ourexpv28reproduce.papers.patcnet.preparation.READERS") as readers:
            readers["bci2a"].read_trials.return_value = trials
            x, y = load_subject("unused", "bci2a", 1)
        self.assertEqual(x.shape, (20, 22, 500))
        self.assertTrue(0.1 < x.std() < 100)
        self.assertEqual(compute_roi_features(x, channel_info("bci2a"))["roi_vector"].shape, (20, 2000))

    def test_within_subject_folds_and_reporting(self):
        from ..papers.patcnet import experiment, command
        from ..evaluation.figures import folds_of, pooled
        x = np.zeros((40, 29, 475), dtype=np.float32)
        groups = np.repeat(np.arange(20), 2)
        y = np.array([0, 1] * 20)
        def fake_fold(xs, ys, xt, yt, *args):
            self.assertEqual((len(xs), len(xt)), (36, 4))
            h = [dict(loss=1., accuracy=1., predictions=yt.tolist(), scores=yt.tolist())]
            return dict(fold_scores(h, yt.tolist(), 30), labels=yt.tolist(),
                        training_seconds=1., peak_gpu_memory_bytes=None)
        with patch.object(experiment, "load_comparison_subject", return_value=(x, y, groups)), \
             patch.object(experiment, "run_fold", side_effect=fake_fold), \
             patch.object(experiment, "model_complexity", return_value={}):
            result = experiment.run("unused", "liu2024", held_out=[1], protocol="within", show_progress=False)
        folds = result["results"]["1"]["folds"]
        self.assertEqual(len(folds_of(result)), 10)
        self.assertEqual(sorted(i for f in folds for i in f["test_index"]), list(range(40)))
        from ..datascripts.version27 import fold_indices, SUBJECTS
        self.assertEqual(result["cohort_subjects"], list(SUBJECTS["liu2024"]))
        self.assertEqual(result["cohort"], "version27")
        self.assertEqual(result["class_names"], ["rest", "imagery"])
        for f, (train, test) in zip(folds, fold_indices(groups, y)):
            self.assertEqual(f["train_index"], train.tolist())
            self.assertEqual(f["test_index"], test.tolist())
            self.assertFalse(set(groups[train]) & set(groups[test]))
            self.assertFalse(set(f["train_index"]) & set(f["test_index"]))
        self.assertEqual(len(pooled(result, "labels")), 40)
        self.assertEqual(result["cost"]["training_seconds_total"], 10.)
        self.assertEqual(result["summary"]["final_epoch"]["accuracy"], 1.)
        self.assertTrue(command.summary_text(result))

    def test_scores_follow_the_published_protocol(self):
        # Loss stops improving after epoch 2; with patience 3 the code stops at epoch 5.
        self.assertEqual(stopping_epoch([1.0, 0.8, 0.9, 0.9, 0.9, 0.7, 0.6], patience=3), 5)
        self.assertEqual(stopping_epoch([1.0, 0.9, 0.8], patience=3), 3)
        labels = [0, 1, 0, 1]
        history = [dict(loss=l, accuracy=a, predictions=p, scores=[0.5] * 4) for l, a, p in (
            (1.0, 0.50, [0, 0, 0, 1]), (0.8, 0.75, [0, 1, 0, 0]), (0.9, 0.50, [1, 1, 1, 1]), (0.9, 0.50, [1, 1, 1, 1]),
            (0.9, 0.50, [1, 1, 1, 1]), (0.7, 1.00, [0, 1, 0, 1]))]
        result = fold_scores(history, labels, patience=3)
        # The epoch with 100% comes after the code would have stopped: the published score is 75% (epoch 2).
        self.assertEqual((result["as_published"]["epoch"], result["as_published"]["stopping_epoch"]), (2, 5))
        self.assertEqual(result["as_published"]["accuracy"], 0.75)
        self.assertEqual((result["final_epoch"]["epoch"], result["final_epoch"]["accuracy"]), (6, 1.0))

    def test_a_short_fold_runs(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(24, 30, 1000)).astype(np.float32)
        y = np.array([0, 1] * 12)
        from dataclasses import replace
        settings = replace(DATASETS["liu2024"], warmup_epochs=1)
        result = run_fold(x[:16], y[:16], x[16:], y[16:], channel_info("liu2024"), settings, torch.device("cpu"),
                          training=TrainingSettings(epochs=2, batch_size=8), show_progress=False)
        self.assertEqual(len(result["test_accuracy_per_epoch"]), 2)
        self.assertEqual(result["final_epoch"]["accuracy"], result["test_accuracy_per_epoch"][-1])
        jia = replace(DATASETS["jia2019"], warmup_epochs=1)          # confidence-only PGTC, without channel names
        xj = rng.normal(size=(12, 63, 1708)).astype(np.float32)
        result = run_fold(xj[:8], y[:8], xj[8:], y[8:12], None, jia, torch.device("cpu"),
                          training=TrainingSettings(epochs=2, batch_size=4), show_progress=False)
        self.assertEqual(len(result["calibrated_fraction_per_epoch"]), 2)
        # Recorded for reporting: training curves, calibration accuracy, per-trial scores for ROC, time.
        for key in ("train_loss_per_epoch", "train_accuracy_per_epoch", "test_loss_per_epoch",
                    "target_accuracy_per_epoch"):
            self.assertEqual(len(result[key]), 2, key)
        for key in ("final_epoch", "as_published"):
            self.assertEqual(len(result[f"{key}_predictions"]), 4)
            self.assertEqual(len(result[f"{key}_scores"]), 4)
        self.assertEqual(result["final_epoch_predictions"], (np.array(result["final_epoch_scores"]) > 0.5).astype(int).tolist())
        self.assertGreater(result["training_seconds"], 0)

    def test_bci_pgtc_fold_keeps_all_test_trials(self):
        from dataclasses import replace
        rng = np.random.default_rng(3)
        x = rng.normal(size=(25, 22, 500)).astype(np.float32)
        y = np.arange(25) % 2
        result = run_fold(x[:16], y[:16], x[16:], y[16:], channel_info("bci2a"),
                          replace(DATASETS["bci2a"], warmup_epochs=1), torch.device("cpu"),
                          training=TrainingSettings(epochs=2, batch_size=8), show_progress=False)
        self.assertEqual(len(result["labels"]), 9)
        self.assertEqual(len(result["final_epoch_scores"]), 9)
        self.assertTrue(np.isfinite(result["test_loss_per_epoch"]).all())

    def test_comparison_subjects_and_model_shape(self):
        from ..datascripts.version27 import SUBJECTS
        from ..papers.patcnet.experiment import subjects_of
        for dataset, count in (("liu2024", 46), ("bci2a", 9), ("jia2019", 15)):
            self.assertEqual(subjects_of(dataset, "version27"), SUBJECTS[dataset])
            self.assertEqual(len(SUBJECTS[dataset]), count)
        settings = dataset_settings("liu2024")
        model = build_model("liu2024", settings=settings).eval()
        with torch.no_grad():
            self.assertEqual(model(torch.randn(3, 29, 475)).shape, (3, 2))
        info = channel_info("liu2024", "version27")
        self.assertEqual(len(info["channel_names"]), 29)
        self.assertEqual(info["roi_groups"]["midline"], ["FCz", "Cz"])
        self.assertEqual(len(subjects_of("liu2024", "published")), 24)
        self.assertEqual(len(subjects_of("liu2024", "distinct")), 23)

    def test_real_comparison_liu_samples_and_groups(self):
        if not liu2024.subject_file(LIU, 2).is_file():
            self.skipTest("Liu recordings unavailable")
        from ..datascripts.version27 import read_samples, fold_indices
        samples = read_samples(LIU, "liu2024", 2)
        x, y, groups = load_comparison_subject(LIU, "liu2024", 2)
        self.assertEqual(x.shape, (len(samples), 29, 475))
        np.testing.assert_array_equal(y, [s.label for s in samples])
        np.testing.assert_array_equal(groups, [s.group for s in samples])
        self.assertEqual(y.tolist(), [v for _ in range(len(y) // 2) for v in (1, 0)])
        self.assertTrue(np.isfinite(x).all())
        for train, test in fold_indices(groups, y):
            self.assertFalse(set(groups[train]) & set(groups[test]))
        from dataclasses import replace
        settings = replace(dataset_settings("liu2024"), warmup_epochs=1)
        train, test = fold_indices(groups, y)[0]
        record = run_fold(x[train], y[train], x[test], y[test], channel_info("liu2024", "version27"),
                          settings, torch.device("cpu"), training=TrainingSettings(epochs=2, batch_size=8),
                          show_progress=False)
        self.assertEqual(record["labels"], y[test].tolist())
        self.assertTrue(np.isfinite(record["test_loss_per_epoch"]).all())

    def test_complexity(self):
        state = torch.get_rng_state()
        size = model_complexity(lambda: build_model("liu2024"), np.zeros((3, 30, 1000), np.float32),
                                torch.device("cpu"), batch_size=2, repeats=3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))         # training draws are not disturbed
        self.assertEqual(size["parameters"], 71132)
        self.assertGreater(size["flops_per_trial"], 0)
        self.assertEqual(size["latency_batch_size"], 2)
        self.assertGreater(size["model_file_bytes"], 4 * size["parameters"])
        self.assertIn("torch", environment(torch.device("cpu"))["packages"])

    @unittest.skipUnless(liu2024.subject_file(LIU, 2).is_file(), "Liu 2024 recordings are not on this machine")
    def test_real_liu2024(self):
        x, y = load_subject(LIU, "liu2024", 2)
        self.assertEqual((x.shape, int(y.sum())), ((40, 30, 1000), 20))

    @unittest.skipUnless((JIA / "exp1-S1-left.mat").is_file(), "Jia 2019 recordings are not on this machine")
    def test_real_jia2019(self):
        x, y = load_subject(JIA, "jia2019", 1)
        self.assertEqual((x.shape, int(y.sum())), ((80, 63, 1708), 40))


if __name__ == "__main__":
    unittest.main()
