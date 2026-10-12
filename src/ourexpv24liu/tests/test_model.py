"""The classifier and its three paths."""
import unittest

import torch
from torch_geometric.data import Batch, Data

from ..data import bci2a_config, liu2024_config
from ..model import EEGGraphModel, ModelConfig

BCI2A = ModelConfig(num_electrodes=22, signal_samples=500)


def random_graph(generator, n=22, samples=500):
    pairs = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j]).T
    return Data(x=torch.rand(n, 6, generator=generator) + 0.1, edge_index=pairs,
                edge_attr=torch.rand(pairs.size(1), 2, generator=generator),
                signal=torch.randn(n, samples, generator=generator), y=torch.tensor(0))


def call(model, batch):
    return model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.signal)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(0)
        cls.batch = Batch.from_data_list([random_graph(generator) for _ in range(4)])

    def test_size_shape_and_gradients(self):
        torch.manual_seed(0)
        model = EEGGraphModel(BCI2A)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 44822)   # as in version 23
        out = call(model, self.batch)
        self.assertEqual(tuple(out.shape), (4, 2))
        out.sum().backward()
        self.assertEqual([name for name, p in model.named_parameters() if p.grad is None], [])
        self.assertEqual(float(model.wide.weight.detach().abs().sum()), 0.0)  # the wide path starts at zero

    def test_each_path_contributes_and_graphs_do_not_mix(self):
        torch.manual_seed(0)
        model = EEGGraphModel(BCI2A).eval()
        torch.nn.init.normal_(model.wide.weight, std=0.01)
        reference = call(model, self.batch)
        for field in ("x", "edge_attr", "signal"):   # graph head / wide path, pair stream, temporal encoder
            changed = self.batch.clone()
            changed[field] = changed[field] * 1.5
            self.assertFalse(torch.allclose(call(model, changed), reference), field)
        # In evaluation mode a graph's logits do not depend on the other graphs of the batch.
        single = Batch.from_data_list([self.batch.get_example(2)])
        torch.testing.assert_close(call(model, single)[0], reference[2], atol=1e-5, rtol=1e-4)

    def test_requires_the_fixed_electrode_layout(self):
        model = EEGGraphModel(BCI2A)
        with self.assertRaises(ValueError):
            model(self.batch.x[:21], self.batch.edge_index, self.batch.batch[:21], self.batch.edge_attr,
                  self.batch.signal[:21])
        with self.assertRaises(ValueError):
            EEGGraphModel(ModelConfig(hidden=30))

    def test_input_size_follows_the_dataset(self):
        config = ModelConfig.for_data(liu2024_config())
        self.assertEqual((config.num_electrodes, config.signal_samples), (29, 1000))
        self.assertEqual(config, ModelConfig())
        self.assertEqual(ModelConfig.for_data(bci2a_config()), BCI2A)
        generator = torch.Generator().manual_seed(1)
        batch = Batch.from_data_list([random_graph(generator, 29, 1000) for _ in range(3)])
        torch.manual_seed(0)
        out = call(EEGGraphModel(config), batch)
        self.assertEqual(tuple(out.shape), (3, 2))
        with self.assertRaises(ValueError):          # a model built for one dataset rejects the other
            call(EEGGraphModel(BCI2A), batch)


if __name__ == "__main__":
    unittest.main()
