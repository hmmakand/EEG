"""EEGNet: the port of the model, the preparation, one fold, and version 27's samples."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ..datascripts import Trial, liu2024
from ..datascripts.version27 import SUBJECTS, Sample, fold_indices
from ..papers.eegnet.experiment import validation_subjects
from ..papers.eegnet.network import build_model
from ..datascripts.standardize import exponential_moving_standardize
from ..papers.eegnet.preparation import load_subject, prepare_samples
from ..papers.eegnet.settings import ModelSettings, TrainingSettings
from ..papers.eegnet.training import SCORES, class_weights, run_fold
from ..run import main

LIU = Path(os.environ.get("LIU2024_DATA", "/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data"))


class EEGNetTests(unittest.TestCase):
    def test_model_matches_the_paper(self):
        # Table 3: EEGNet-8,2 on the SMR data (22 channels, 2 s at 128 Hz, 4 classes, kernel 32) has 1,716
        # trainable parameters.
        model = build_model(22, 256, 0.25, classes=4)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1716)
        self.assertEqual(tuple(model.eval()(torch.randn(3, 22, 256)).shape), (3, 4))
        self.assertEqual(ModelSettings().kernel, 32)
        # The max-norm constraints hold after a step.
        with torch.no_grad():
            model.spatial.weight.mul_(10)
            model.dense.weight.mul_(10)
        model.apply_constraints()
        self.assertLessEqual(float(model.spatial.weight.detach().flatten(1).norm(dim=1).max()), 1.0 + 1e-5)
        self.assertLessEqual(float(model.dense.weight.detach().norm(dim=1).max()), 0.25 + 1e-5)

    def test_class_weights_follow_the_paper(self):
        self.assertIsNone(class_weights(np.array([0, 1] * 5)))
        weights = class_weights(np.array([0] * 56 + [1] * 10))
        assert weights is not None
        self.assertEqual(weights.tolist(), [1.0, 6.0])   # 5.6:1 -> 6

    def test_preparation(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(3000, 4)) * 5 + 2
        standardized = exponential_moving_standardize(data, 1e-3, 1000, 1e-4)
        np.testing.assert_allclose(standardized[:1000].mean(axis=0), 0, atol=1e-8)
        np.testing.assert_allclose(standardized[:1000].std(axis=0), 1, atol=1e-6)
        trials = [Trial(rng.normal(size=(2000, 29)) * 1e-5, 500, 0, 1, i) for i in range(2)]
        samples = [Sample(t, t.onset + start, 475, label, t.trial_id) for t in trials
                   for start, label in ((250, 1), (1000, 0))]
        x, y, groups = prepare_samples(samples)
        self.assertEqual((x.shape, x.dtype, y.tolist(), groups.tolist()), ((4, 29, 243), np.float32, [1, 0] * 2,
                                                                          [0, 0, 1, 1]))

    def test_a_short_fold_and_its_scores(self):
        rng = np.random.default_rng(1)
        make = lambda n: (rng.normal(size=(n, 8, 64)).astype(np.float32), np.array([0, 1] * (n // 2)))
        record, state = run_fold(make(32), make(8), make(10), 0.25, torch.device("cpu"),
                                 TrainingSettings(epochs=4, batch_size=8), show_progress=False)
        for name in SCORES:
            self.assertEqual(len(record[f"{name}_scores"]), 10)
        losses = record["validation_loss_per_epoch"]
        self.assertEqual(record["validation_selected"]["epoch"], int(np.argmin(losses)) + 1)
        self.assertEqual(record["final_epoch"]["epoch"], 4)
        self.assertEqual(record["highest_test_epoch"]["accuracy"], max(record["test_accuracy_per_epoch"]))
        self.assertEqual(len(record["train_loss_per_epoch"]), 4)
        assert state is not None
        self.assertIn("separable_pointwise.weight", state)

    def test_folds_and_validation_subjects(self):
        groups = [t for t in range(20) for _ in (0, 1)]          # two samples per trial, as liu2024
        folds = fold_indices(groups, [1, 0] * 20)
        self.assertEqual(len(folds), 10)
        self.assertEqual(sorted(int(i) for _, test in folds for i in test), list(range(40)))
        others = [s for s in SUBJECTS["liu2024"] if s != 2]
        chosen = validation_subjects(others, 2, TrainingSettings())
        self.assertEqual((len(SUBJECTS["liu2024"]), len(chosen)), (46, 17))
        self.assertEqual(chosen, validation_subjects(others, 2, TrainingSettings()))     # fixed by the seed
        self.assertEqual(len(validation_subjects([1, 2, 3, 5, 6, 7, 8], 9, TrainingSettings())), 3)

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(SystemExit):
            main(["--paper", "patcnet", "--dataset", "jia2019", "--data-dir", folder, "--save-models"])
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(SystemExit):
            main(["--paper", "eegnet", "--dataset", "bci2a", "--data-dir", folder, "--subjects", "10"])

    @unittest.skipUnless(liu2024.subject_file(LIU, 2).is_file(), "Liu 2024 recordings are not on this machine")
    def test_real_liu2024(self):
        x, y, groups = load_subject(LIU, "liu2024", 2)
        self.assertEqual((x.shape, int(y.sum()), len(set(groups.tolist()))), ((80, 29, 243), 40, 40))


if __name__ == "__main__":
    unittest.main()
