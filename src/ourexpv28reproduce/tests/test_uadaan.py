"""UA-DAAN: the port against the reference implementation, the DE features, evaluation without side effects."""
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
from scipy import signal
import torch

from ..datascripts import SAMPLE_RATE, Trial
from ..datascripts.version27 import Sample
from ..papers.uadaan.features import differential_entropy
from ..papers.uadaan.network import build_model
from ..papers.uadaan.settings import FeatureSettings, TrainingSettings
from ..papers.uadaan.training import SCORES, mc_predict, run_fold

ORIGINAL = Path(__file__).parents[1] / "papers" / "uadaan" / "original"


class UADAANTests(unittest.TestCase):
    def test_port_trains_as_the_reference(self):
        import matplotlib
        matplotlib.use("Agg")
        with mock.patch.dict(sys.modules, seaborn=types.ModuleType("seaborn")), \
                mock.patch.object(sys, "path", [str(ORIGINAL), *sys.path]):
            import model as reference   # pyright: ignore[reportMissingImports]  (from ORIGINAL, put on sys.path above)
            torch.manual_seed(0)
            original = reference.Model()
        for name in ("model", "bayesian", "classifier", "extractor", "discriminator"):
            sys.modules.pop(name, None)
        port = build_model(640)
        self.assertEqual([tuple(p.shape) for p in original.parameters()], [tuple(p.shape) for p in port.parameters()])
        with torch.no_grad():
            for source, target in zip(original.parameters(), port.parameters()):
                target.copy_(source)
        x, t, y = torch.randn(20, 640), torch.randn(8, 640), torch.randint(0, 2, (20,))
        for step in range(2):
            torch.manual_seed(step)
            c1, t1 = original(x, t, y)
            torch.manual_seed(step)
            c2, t2 = port(x, t, y)
            torch.testing.assert_close(c2.reshape(-1), c1.reshape(-1))
            torch.testing.assert_close(t2, t1)
            g1 = torch.autograd.grad((c1 - t1).sum(), list(original.parameters()), allow_unused=True)
            g2 = torch.autograd.grad((c2 - t2).sum(), list(port.parameters()), allow_unused=True)
            for a, b in zip(g1, g2):
                if a is not None:
                    torch.testing.assert_close(b, a)
        original.epoch_based_processing(2)
        port.adapt_loss.update_dynamic(2)
        self.assertAlmostEqual(port.adapt_loss.dynamic_factor, original.adapt_loss.dynamic_factor)

    def test_differential_entropy(self):
        rng = np.random.default_rng(0)
        trial = Trial(rng.normal(size=(2000, 3)) * 1e-5, 0, 1, 1, 0)
        features, labels, _ = differential_entropy([Sample(trial, 500, 475, 1, 0)], "liu2024")
        self.assertEqual(features.shape, (1, 3 * 5))
        sos = signal.butter(4, (8, 13), "bandpass", fs=SAMPLE_RATE, output="sos")
        alpha = signal.sosfiltfilt(sos, trial.signal, axis=0)[500:975, 1]
        self.assertAlmostEqual(float(features[0, 1 * 5 + 2]), 0.5 * np.log(2 * np.pi * np.e * alpha.var()), places=4)
        self.assertEqual(FeatureSettings().bands[-1], (30, 40))

    def test_scoring_does_not_change_training(self):
        model, x = build_model(10), torch.randn(4, 10)
        state = torch.get_rng_state()
        mc_predict(model, x, torch.device("cpu"))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_a_short_fold(self):
        rng = np.random.default_rng(1)
        make = lambda n: (rng.normal(size=(n, 12)).astype(np.float32), np.array([0, 1] * (n // 2)))
        record, state = run_fold(make(40), make(10), torch.device("cpu"), TrainingSettings(epochs=3),
                                 show_progress=False)
        for name in SCORES:
            self.assertEqual(len(record[f"{name}_scores"]), 10)
        self.assertEqual(record["final_epoch"]["epoch"], 3)
        self.assertEqual(len(record["transfer_loss_per_epoch"]), 3)
        self.assertTrue(0.0 <= record["dynamic_factor_final"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
