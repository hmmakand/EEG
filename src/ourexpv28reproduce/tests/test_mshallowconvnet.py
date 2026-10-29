"""M-ShallowConvNet: the port of the model, CutCat, bagging, preparation and one fold."""
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from ..datascripts import Trial, liu2024
from ..datascripts.version27 import Sample
from ..datascripts.splits import split_groups
from ..papers.mshallowconvnet.network import build_model
from ..papers.mshallowconvnet.preparation import load_subject, prepare_samples
from ..papers.mshallowconvnet.settings import TrainingSettings
from ..papers.mshallowconvnet.training import SCORES, bagged, cutcat, run_fold

LIU = Path(os.environ.get("LIU2024_DATA", "/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data"))


class MShallowConvNetTests(unittest.TestCase):
    def test_model_matches_the_code(self):
        # The published Net(4 classes, 22 channels, 250 Hz) on 3 s epochs (751 samples) has 16,468 parameters.
        model = build_model(22, 751, classes=4)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 16468)
        self.assertLessEqual(float(model.dense.weight.detach().norm(dim=1).max()), 0.5 + 1e-6)   # renormed once
        with torch.no_grad():
            model.temporal.weight.mul_(100)
        model.eval()(torch.randn(2, 22, 751))
        self.assertLessEqual(float(model.temporal.weight.detach().flatten(1).norm(dim=1).max()), 2.0 + 1e-5)
        self.assertEqual((model.temporal.kernel_size, model.pool.kernel_size, model.pool.stride),
                         ((1, 30), (1, 75), (1, 22)))

    def test_cutcat(self):
        torch.manual_seed(0)
        x, y = torch.randn(16, 3, 400), torch.tensor([0, 1] * 8)
        mixed, labels = cutcat(x, y, 0.125)
        self.assertEqual(mixed.shape, x.shape)
        torch.testing.assert_close(labels.sum(dim=1), torch.ones(16))
        self.assertTrue(bool((labels[torch.arange(16), y] >= 1 - 0.125).all()))      # at most lambda replaced
        changed = (mixed != x).any(dim=1).float().mean(dim=1)
        torch.testing.assert_close(changed, 1 - labels[torch.arange(16), y])         # labels follow the share
        same, labels = cutcat(x[:4], torch.zeros(4, dtype=torch.long), 0.125)          # no partner of another class
        self.assertTrue(torch.equal(same, x[:4]))
        self.assertEqual(labels[:, 0].tolist(), [1.0] * 4)

    def test_bagging_votes(self):
        probabilities = torch.tensor([[[0.6, 0.4]], [[0.45, 0.55]], [[0.4, 0.6]]])     # two of three say class 1
        predictions, scores = bagged(probabilities)
        self.assertEqual(predictions, [1])
        self.assertAlmostEqual(scores[0], (0.4 + 0.55 + 0.6) / 3, places=6)

    def test_preparation_and_splits(self):
        rng = np.random.default_rng(0)
        trials = [Trial(rng.normal(size=(2000, 29)) * 1e-5, 500, 0, 1, i) for i in range(2)]
        samples = [Sample(t, t.onset + start, 475, label, t.trial_id) for t in trials
                   for start, label in ((250, 1), (1000, 0))]
        x, y, groups = prepare_samples(samples)
        self.assertEqual(x.shape, (4, 29, 475))
        np.testing.assert_allclose(x.mean(axis=2), 0, atol=1e-5)           # each sample standardised on its own
        parts = split_groups(list(range(1, 46)), 5, 42, 2)
        self.assertEqual(sorted(s for part in parts for s in part), list(range(1, 46)))
        self.assertEqual([len(p) for p in parts], [9] * 5)

    def test_a_short_fold(self):
        rng = np.random.default_rng(1)
        make = lambda n: (rng.normal(size=(n, 4, 100)).astype(np.float32), np.array([0, 1] * (n // 2)))
        splits = [(make(16), make(6)) for _ in range(5)]
        x_test, y_test = make(10)
        record, states = run_fold(splits, x_test, y_test, 0.1, torch.device("cpu"),
                                  TrainingSettings(epochs=3), show_progress=False)
        for name in SCORES:
            self.assertEqual(len(record[f"{name}_scores"]), 10)
        self.assertEqual(len(record["validation_selected"]["epochs"]), 5)
        self.assertEqual(record["highest_test_epoch"]["accuracy"], max(record["ensemble_test_accuracy_per_epoch"]))
        self.assertEqual((len(states), len(record["train_loss_per_epoch"])), (5, 3))

    @unittest.skipUnless(liu2024.subject_file(LIU, 2).is_file(), "Liu 2024 recordings are not on this machine")
    def test_real_liu2024(self):
        x, y, groups = load_subject(LIU, "liu2024", 2)
        self.assertEqual((x.shape, int(y.sum())), ((80, 29, 475), 40))


if __name__ == "__main__":
    unittest.main()
