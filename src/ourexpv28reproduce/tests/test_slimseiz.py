"""SlimSeiz: the port against the notebook's model, the contrastive loss, channel selection and one fold."""
import json
import math
from pathlib import Path
from typing import Any
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..papers.slimseiz.channels import select_channels, smote
from ..papers.slimseiz.network import build_model
from ..papers.slimseiz.preparation import SCALE, prepare_samples
from ..papers.slimseiz.settings import ChannelSelectionSettings, TrainingSettings
from ..papers.slimseiz.training import SCORES, ntxent, run_fold
from ..datascripts import Trial
from ..datascripts.version27 import Sample

NOTEBOOK = Path(__file__).parents[1] / "papers" / "slimseiz" / "original" / "Loop_ADDNet_kFold.ipynb"


class SlimSeizTests(unittest.TestCase):
    def test_port_computes_the_notebook_model(self):
        from einops import einsum, rearrange, repeat
        cells = json.loads(NOTEBOOK.read_text())["cells"]
        namespace: dict[str, Any] = dict(torch=torch, nn=nn, F=F, math=math, rearrange=rearrange, repeat=repeat, einsum=einsum)
        for index in (1, 7):                                      # MambaBlock and RMSNorm; OneDCNN
            exec("".join(cells[index]["source"]), namespace)
        torch.manual_seed(0)
        original, port = namespace["OneDCNN"](8), build_model(8)
        self.assertEqual([tuple(p.shape) for p in original.parameters()], [tuple(p.shape) for p in port.parameters()])
        with torch.no_grad():
            for source, target in zip(original.parameters(), port.parameters()):
                target.copy_(source)
        x = torch.randn(3, 8, 1024)
        for theirs, ours in zip(original.eval()(x), port.eval()(x)):
            torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)

    def test_ntxent_follows_the_library_definition(self):
        # Two samples per class: every sample has one positive and two negatives.
        e = torch.tensor([[1., 0.], [1., 0.1], [0., 1.], [0.1, 1.]])
        y = torch.tensor([0, 0, 1, 1])
        s = F.normalize(e, dim=1) @ F.normalize(e, dim=1).T / 0.5
        expected = np.mean([-(s[i, j] - torch.log(torch.exp(s[i, j]) + torch.exp(s[i, [k for k in range(4) if y[k] != y[i]]]).sum()))
                            for i in range(4) for j in range(4) if i != j and y[i] == y[j]])
        self.assertAlmostEqual(float(ntxent(e, y, 0.5)), float(expected), places=5)
        self.assertEqual(float(ntxent(e[:2], torch.tensor([0, 1]), 0.5)), 0.0)          # no positive pair

    def test_preparation_scale_and_smote(self):
        trial = Trial(np.full((500, 3), 2e-5), 0, 1, 1, 0)
        x, _, _ = prepare_samples([Sample(trial, 0, 250, 1, 0)], "liu2024")
        self.assertAlmostEqual(float(x[0, 0, 0]), 2e-5 * SCALE, places=6)                  # volts x 1e4
        x, _, _ = prepare_samples([Sample(Trial(np.full((500, 3), 20.0), 0, 1, 1, 0), 0, 250, 1, 0)], "bci2a")
        self.assertAlmostEqual(float(x[0, 0, 0]), 20e-6 * SCALE, places=6)                 # microvolts -> volts first
        rng = np.random.default_rng(0)
        features, labels = smote(rng.normal(size=(13, 4)), np.array([0] * 10 + [1] * 3), 5, rng)
        self.assertEqual(np.bincount(labels).tolist(), [10, 10])

    def test_channel_selection_finds_the_informative_channels(self):
        rng = np.random.default_rng(1)
        y = np.array([0, 1] * 60)
        x = rng.normal(size=(120, 6, 64))
        x[:, 1] += y[:, None] * 2.0                                                        # channels 1 and 4 carry
        x[:, 4] -= y[:, None] * 2.0                                                        # the class
        chosen, counts = select_channels(x, y, ChannelSelectionSettings(channels=2, repetitions=5, components=10),
                                         seed=3)
        self.assertEqual(chosen, [1, 4])
        self.assertEqual(sum(counts.values()), 2 * 5)

    def test_a_short_fold(self):
        rng = np.random.default_rng(2)
        make = lambda n: (rng.normal(size=(n, 3, 256)).astype(np.float32), np.array([0, 1] * (n // 2)))
        record, state = run_fold(make(24), make(8), make(6), torch.device("cpu"), TrainingSettings(epochs=3),
                                 show_progress=False)
        for name in SCORES:
            self.assertEqual(len(record[f"{name}_scores"]), 6)
        self.assertEqual(record["validation_selected"]["epoch"],
                         int(np.argmax(record["validation_accuracy_per_epoch"])) + 1)
        self.assertIn("mixer.A_log", state)


if __name__ == "__main__":
    unittest.main()
