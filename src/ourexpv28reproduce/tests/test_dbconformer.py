"""DBConformer: the port against the published model, Euclidean alignment (offline and online), one fold."""
import argparse
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
from scipy.linalg import fractional_matrix_power
import torch

from ..papers.dbconformer.network import build_model
from ..papers.dbconformer.preparation import align, align_online
from ..papers.dbconformer.settings import DEPTH, TrainingSettings
from ..papers.dbconformer.training import SCORES, run_fold

ORIGINAL = Path(__file__).parents[1] / "papers" / "dbconformer" / "original" / "models"


def stub_timm():
    """The original imports timm only for trunc_normal_, which torch has too."""
    layers = types.ModuleType("timm.models.layers")
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    models = types.ModuleType("timm.models")
    models.layers = layers
    timm = types.ModuleType("timm")
    timm.models = models
    return {"timm": timm, "timm.models": models, "timm.models.layers": layers}


class DBConformerTests(unittest.TestCase):
    def test_port_computes_the_published_model(self):
        torch.manual_seed(0)
        with mock.patch.dict(sys.modules, stub_timm()), mock.patch.object(sys, "path", [str(ORIGINAL), *sys.path]):
            import DBConformer as published   # pyright: ignore[reportMissingImports]  (from ORIGINAL, put on sys.path above)
            args = argparse.Namespace(data_name="MI1-7", chn=22, patch_size=125, time_sample_num=750, class_num=2,
                                      spa_dim=16, gate_flag=False, posemb_flag=True, branch="all",
                                      chn_atten_flag=True)
            original = published.DBConformer(args, emb_size=40, tem_depth=2, chn_depth=2, chn=22, n_classes=2)
        sys.modules.pop("DBConformer", None)
        port = build_model(22, 750, depth=2)
        self.assertEqual([tuple(p.shape) for p in original.parameters()], [tuple(p.shape) for p in port.parameters()])
        with torch.no_grad():
            for source, target in zip(original.parameters(), port.parameters()):
                target.copy_(source)
        x = torch.randn(3, 22, 750)
        torch.testing.assert_close(port.eval()(x), original.eval()(x.unsqueeze(1))[1], atol=1e-5, rtol=1e-5)

    def test_alignment_follows_the_code(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(6, 4, 200)) * np.array([1, 2, 3, 4])[None, :, None]
        reference = fractional_matrix_power(np.mean([np.cov(t) for t in x], axis=0), -0.5)       # alg_utils.EA
        np.testing.assert_allclose(align(x), np.stack([reference @ t for t in x]), atol=1e-8)
        online = align_online(x)
        for k in range(len(x)):                         # utils.data_loader: the running mean of the covariances
            r = fractional_matrix_power(np.mean([np.cov(t) for t in x[:k + 1]], axis=0), -0.5)
            np.testing.assert_allclose(online[k], r @ x[k], atol=1e-8)
        referenced = x - x.mean(axis=1, keepdims=True)  # rank-deficient (common average reference): still finite
        self.assertTrue(np.isfinite(align(referenced)).all() and np.isfinite(align_online(referenced)).all())

    def test_depth_and_a_short_fold(self):
        self.assertEqual((DEPTH["loso"]["bci2a"], DEPTH["loso"]["liu2024"], DEPTH["within"]["jia2019"]), (2, 6, 2))
        rng = np.random.default_rng(1)
        make = lambda n: (rng.normal(size=(n, 4, 250)), np.array([0, 1] * (n // 2)))
        record, state = run_fold(make(20), make(6), 2, torch.device("cpu"), True, TrainingSettings(epochs=3, batch_size=8),
                                 show_progress=False)
        for name in SCORES:
            self.assertEqual(len(record[f"{name}_scores"]), 6)
        self.assertEqual(record["final_epoch"]["epoch"], 3)
        self.assertEqual(record["highest_test_epoch"]["accuracy"], max(record["test_accuracy_per_epoch"]))
        self.assertIn("channel_attention.0.weight", state)


if __name__ == "__main__":
    unittest.main()
