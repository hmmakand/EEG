"""MSCFormer: the port against the published model, preparation, one fold and the averaging of the five models."""
import sys
from pathlib import Path
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from ..papers.mscformer.experiment import subject_scores
from ..papers.mscformer.network import DepthwiseSpatialConv, build_model
from ..papers.mscformer.preparation import z_score
from ..papers.mscformer.settings import ModelSettings, TrainingSettings
from ..papers.mscformer.training import SCORES, run_fold

ORIGINAL = Path(__file__).parents[1] / "papers" / "mscformer" / "original"


class MSCFormerTests(unittest.TestCase):
    def test_port_computes_the_published_model(self):
        torch.manual_seed(0)
        with mock.patch.dict(sys.modules, torchsummary=types.SimpleNamespace(summary=None)), \
                mock.patch.object(sys, "path", [str(ORIGINAL), *sys.path]), \
                mock.patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self):   # the code calls .cuda()
            import MSCFormer_model as published   # pyright: ignore[reportMissingImports]  (from ORIGINAL, put on sys.path above)
            parameters = published.Parameters(0.5)
            parameters.pooling_size = 44                                            # BCI IV 2a
            original = published.MSCFormer(parameters, database_type="A")
            port = build_model(22, 1000, classes=4)
            named = list(port.named_parameters())
            ours = ([p for n, p in named if n.startswith("branches")] + [port.position]
                    + [p for n, p in named if not n.startswith(("branches", "position"))])
            theirs = list(original.parameters())
            self.assertEqual([tuple(p.shape) for p in theirs], [tuple(p.shape) for p in ours])
            with torch.no_grad():
                for source, target in zip(theirs, ours):
                    target.copy_(source)
            x = torch.randn(3, 22, 1000)
            torch.testing.assert_close(port.eval()(x), original.eval()(x.unsqueeze(1))[1], atol=1e-5, rtol=1e-5)
        sys.modules.pop("MSCFormer_model", None)

    def test_depthwise_spatial_convolution_equals_conv2d(self):
        torch.manual_seed(1)
        ours = DepthwiseSpatialConv(4, 4, (5, 1), groups=4)
        theirs = nn.Conv2d(4, 4, (5, 1), groups=4)
        theirs.load_state_dict(ours.state_dict())
        x = torch.randn(2, 4, 5, 30)
        torch.testing.assert_close(ours(x), theirs(x), atol=1e-5, rtol=1e-5)

    def test_settings_and_preparation(self):
        self.assertEqual((ModelSettings().pool, ModelSettings().kernels, TrainingSettings().batch_size
                          * (1 + TrainingSettings().augmentation_factor)), (44, (85, 65, 45), 288))
        train, test = z_score(np.arange(12, dtype=np.float32).reshape(2, 2, 3), np.zeros((1, 2, 3), np.float32))
        self.assertAlmostEqual(float(train.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(train.std()), 1.0, places=5)
        with self.assertRaises(ValueError):                    # the learned position embedding has 100 places
            build_model(22, 44 * 100)

    def test_a_short_fold_and_the_averaging_of_the_models(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(20, 4, 100)).astype(np.float32)
        y = np.array([0, 1] * 10)
        splits = [(np.setdiff1d(np.arange(20), part), part) for part in np.array_split(np.arange(20), 5)]
        x_test, y_test = rng.normal(size=(6, 4, 100)).astype(np.float32), np.array([0, 1] * 3)
        record, states = run_fold((x, y), x_test, y_test, splits, 0.5, torch.device("cpu"),
                                  TrainingSettings(epochs=2, batch_size=8), ModelSettings(pool=10),
                                  show_progress=False)
        self.assertEqual((len(record["members"]), len(states), len(record["train_loss_per_epoch"])), (5, 5, 2))
        scores = subject_scores([record])
        for name in SCORES:
            per_model = scores[name]["accuracy_per_model"]
            self.assertEqual(len(per_model), 5)
            self.assertAlmostEqual(scores[name]["accuracy"], float(np.mean(per_model)))      # the paper's average
            self.assertEqual(len(scores[f"{name}_scores"]), 6)                               # for the ROC curve


if __name__ == "__main__":
    unittest.main()
