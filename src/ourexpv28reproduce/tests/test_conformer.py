"""EEG Conformer: the port against the published model, S&R augmentation, preparation and one fold."""
import os
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from ..datascripts import Trial, liu2024
from ..datascripts.version27 import Sample
from ..papers.conformer.network import SpatialConv, build_model
from ..papers.conformer.preparation import load_subject, prepare_samples, z_score
from ..papers.conformer.settings import TrainingSettings
from ..datascripts.augmentation import segment_and_reconstruct
from ..papers.conformer.training import SCORES, run_fold

LIU = Path(os.environ.get("LIU2024_DATA", "/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data"))
ORIGINAL = Path(__file__).parents[1] / "papers" / "conformer" / "original" / "conformer.py"


def original_conformer():
    """The published Conformer class, without the script's training code and its imports."""
    source = ORIGINAL.read_text()
    namespace = {}
    exec("import math\nimport torch\nimport torch.nn.functional as F\nfrom torch import nn, Tensor\n"
         "from einops import rearrange, reduce, repeat\nfrom einops.layers.torch import Rearrange, Reduce\n"
         + source[source.index("# Convolution module"):source.index("class ExP")], namespace)
    return namespace["Conformer"]()


class ConformerTests(unittest.TestCase):
    def test_port_computes_the_published_model(self):
        torch.manual_seed(0)
        original, port = original_conformer(), build_model(22, 1000, classes=4)
        weights = [p for n, p in original.named_parameters() if not n.startswith("2.clshead")]   # clshead is unused
        self.assertEqual([tuple(p.shape) for p in weights], [tuple(p.shape) for p in port.parameters()])
        with torch.no_grad():
            for source, target in zip(weights, port.parameters()):
                target.copy_(source)
        x = torch.randn(3, 22, 1000)
        torch.testing.assert_close(port.eval()(x), original.eval()(x.unsqueeze(1))[1], atol=1e-5, rtol=1e-5)

    def test_spatial_convolution_equals_conv2d(self):
        torch.manual_seed(1)
        spatial = SpatialConv(4, 4, (5, 1))
        conv = nn.Conv2d(4, 4, (5, 1))
        conv.load_state_dict(spatial.state_dict())
        x = torch.randn(2, 4, 5, 30)
        torch.testing.assert_close(spatial(x), conv(x), atol=1e-5, rtol=1e-5)

    def test_segment_and_reconstruct(self):
        torch.manual_seed(2)
        x = torch.arange(6, dtype=torch.float).view(6, 1, 1).repeat(1, 2, 16)      # sample i is filled with i
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        new_x, new_y = segment_and_reconstruct(x, y, 8, segments=8)
        self.assertEqual((tuple(new_x.shape), sorted(new_y.tolist())), ((8, 2, 16), [0] * 4 + [1] * 4))
        for sample, label in zip(new_x, new_y):
            sources = sample[0, ::2].long()                                        # one value per segment
            self.assertTrue(all(int(y[s]) == int(label) for s in sources))         # every segment from its class

    def test_preparation(self):
        rng = np.random.default_rng(0)
        trials = [Trial(rng.normal(size=(2000, 29)) * 1e-5, 500, 0, 1, i) for i in range(2)]
        samples = [Sample(t, t.onset + start, 475, label, t.trial_id) for t in trials
                   for start, label in ((250, 1), (1000, 0))]
        x, y, groups = prepare_samples(samples)
        self.assertEqual((x.shape, y.tolist()), ((4, 29, 475), [1, 0, 1, 0]))
        train, test = z_score(x[:2], x[2:])
        self.assertAlmostEqual(float(train.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(train.std()), 1.0, places=4)

    def test_a_short_fold(self):
        rng = np.random.default_rng(1)
        make = lambda n: (rng.normal(size=(n, 4, 200)).astype(np.float32), np.array([0, 1] * (n // 2)))
        record, state = run_fold(make(16), make(6), torch.device("cpu"), TrainingSettings(epochs=3, batch_size=8),
                                 show_progress=False)
        for name in SCORES:
            self.assertEqual(len(record[f"{name}_scores"]), 6)
        self.assertEqual(record["final_epoch"]["epoch"], 3)
        self.assertEqual(record["highest_test_epoch"]["accuracy"], max(record["test_accuracy_per_epoch"]))
        self.assertAlmostEqual(record["mean_test_accuracy_over_epochs"], float(np.mean(record["test_accuracy_per_epoch"])),
                               places=4)
        self.assertIn("classifier.6.weight", state)

    @unittest.skipUnless(liu2024.subject_file(LIU, 2).is_file(), "Liu 2024 recordings are not on this machine")
    def test_real_liu2024(self):
        x, y, groups = load_subject(LIU, "liu2024", 2)
        self.assertEqual((x.shape, int(y.sum())), ((80, 29, 475), 40))


if __name__ == "__main__":
    unittest.main()
