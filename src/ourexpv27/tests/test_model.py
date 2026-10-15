"""The classifier and its three paths."""
import unittest

import torch
from torch_geometric.data import Batch, Data

from ..model import EEGGraphModel, ModelConfig


def random_graph(generator, n=29, samples=475):
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
        model = EEGGraphModel()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 64862)
        self.assertEqual(len(model.layers), 3)
        out = call(model, self.batch)
        self.assertEqual(tuple(out.shape), (4, 2))
        out.sum().backward()
        self.assertEqual([name for name, p in model.named_parameters() if p.grad is None], [])
        self.assertEqual(float(model.wide.weight.detach().abs().sum()), 0.0)  # the wide path starts at zero

    def test_each_path_contributes_and_graphs_do_not_mix(self):
        torch.manual_seed(0)
        model = EEGGraphModel().eval()
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
        model = EEGGraphModel()
        with self.assertRaises(ValueError):
            model(self.batch.x[:28], self.batch.edge_index, self.batch.batch[:28], self.batch.edge_attr,
                  self.batch.signal[:28])
        with self.assertRaises(ValueError):
            EEGGraphModel(ModelConfig(hidden=30))

    def test_input_size_follows_the_dataset(self):
        bci2a = ModelConfig(num_electrodes=22, signal_samples=500)
        generator = torch.Generator().manual_seed(1)
        batch = Batch.from_data_list([random_graph(generator, 22, 500) for _ in range(3)])
        torch.manual_seed(0)
        self.assertEqual(tuple(call(EEGGraphModel(bci2a), batch).shape), (3, 2))
        with self.assertRaises(ValueError):          # a model built for one dataset rejects the other
            call(EEGGraphModel(), batch)


if __name__ == "__main__":
    unittest.main()
