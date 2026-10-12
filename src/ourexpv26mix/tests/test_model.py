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
        self.assertEqual(sum(p.numel() for p in model.parameters()), 62294)   # three hybrid blocks
        version23 = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, layers=2))
        self.assertEqual(sum(p.numel() for p in version23.parameters()), 44822)
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

    def test_path_selection(self):
        torch.manual_seed(0)
        full = EEGGraphModel(BCI2A).eval()
        torch.nn.init.normal_(full.wide.weight, std=0.01)
        outputs = {}
        for path in ("graph", "wide", "temporal"):
            part = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, paths=(path,))).eval()
            part.load_state_dict(full.state_dict())
            outputs[path] = call(part, self.batch)
        # The model is the sum of its three paths.
        torch.testing.assert_close(sum(outputs.values()), call(full, self.batch), atol=1e-5, rtol=1e-4)
        # A path that is switched off receives no gradient.
        graph_only = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, paths=("graph",)))
        call(graph_only, self.batch).sum().backward()
        self.assertIsNone(graph_only.temporal.out.weight.grad)
        self.assertIsNone(graph_only.wide.weight.grad)
        self.assertIsNotNone(graph_only.head[0].weight.grad)
        for bad in ((), ("graph", "graph"), ("other",)):
            with self.assertRaises(ValueError):
                ModelConfig(paths=bad)

    def test_node_features_from_the_signal(self):
        torch.manual_seed(0)
        plain = EEGGraphModel(BCI2A)
        sizes = {}
        for mode in ("power", "timed", "mixed"):
            torch.manual_seed(0)
            model = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, node_signal=mode,
                                              paths=("graph",))).eval()
            # The other parameters start as they do without the node encoder.
            torch.testing.assert_close(model.node_encoder.weight, plain.node_encoder.weight)
            torch.testing.assert_close(model.temporal.out.weight, plain.temporal.out.weight)
            sizes[mode] = model.node_signal.out.in_features
            reference = call(model, self.batch)
            self.assertEqual(tuple(reference.shape), (4, 2))
            # The graph head now depends on the signal, which it did not before.
            changed = self.batch.clone()
            changed.signal = changed.signal.flip(1) * 2
            self.assertFalse(torch.allclose(call(model, changed), reference, atol=1e-5), mode)
            model.train()
            call(model, self.batch).sum().backward()
            self.assertGreater(float(model.node_signal.conv.weight.grad.abs().sum()), 0)
        self.assertEqual(sizes, dict(power=8, timed=8 * 29, mixed=2 * 8 * 29))
        # "mixed": the edge features of the trial enter the mixing weights.
        torch.manual_seed(0)
        mixed = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, node_signal="mixed",
                                          paths=("graph",))).eval()
        self.assertEqual(tuple(mixed.node_signal.pair_weight.shape), (2, 22, 22))
        mixed.train()
        call(mixed, self.batch).sum().backward()
        self.assertGreater(float(mixed.node_signal.edge_weight.weight.grad.abs().sum()), 0)
        self.assertGreater(float(mixed.node_signal.pair_weight.grad.abs().sum()), 0)
        # Graphs of a batch do not mix.
        mixed.eval()
        single = Batch.from_data_list([self.batch.get_example(2)])
        torch.testing.assert_close(call(mixed, single)[0], call(mixed, self.batch)[2], atol=1e-4, rtol=1e-3)
        graph_only = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, paths=("graph",))).eval()
        changed = self.batch.clone()
        changed.signal = changed.signal * 2
        torch.testing.assert_close(call(graph_only, changed), call(graph_only, self.batch))
        with self.assertRaises(ValueError):
            ModelConfig(node_signal="other")

    def test_electrode_readout(self):
        torch.manual_seed(0)
        model = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, readout="electrode",
                                          paths=("graph",))).eval()
        self.assertEqual(model.head[0].in_features, 32 * 2 + 22 * 4)
        reference = call(model, self.batch)
        self.assertEqual(tuple(reference.shape), (4, 2))
        # With the mean readout, exchanging two electrodes' final states would not matter; here it does.
        with torch.no_grad():
            model.node_readout.weight.normal_()
        states = torch.randn(88, 32)
        pairs = torch.zeros(4 * 22 * 22, 32)
        batch = self.batch.batch
        pair_graph = torch.arange(4).repeat_interleave(22 * 22)
        swapped = states.clone()
        swapped[[0, 1]] = states[[1, 0]]
        self.assertFalse(torch.allclose(model.graph_logits(states, pairs, batch, pair_graph, 4),
                                        model.graph_logits(swapped, pairs, batch, pair_graph, 4), atol=1e-6))
        plain = EEGGraphModel(BCI2A).eval()
        torch.testing.assert_close(plain.graph_logits(states, pairs, batch, pair_graph, 4),
                                   plain.graph_logits(swapped, pairs, batch, pair_graph, 4), atol=1e-5, rtol=1e-4)
        with self.assertRaises(ValueError):
            ModelConfig(readout="other")

    def test_time_attention_on_the_node_features(self):
        for mode, filters in (("timed", 8), ("mixed", 16)):
            torch.manual_seed(0)
            model = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, node_signal=mode,
                                              node_time_attention=1, paths=("graph",))).eval()
            self.assertEqual(tuple(model.node_signal.time_position.shape), (29, filters))
            out = call(model, self.batch)
            self.assertEqual(tuple(out.shape), (4, 2))
            model.train()
            call(model, self.batch).sum().backward()
            attention = model.node_signal.time_attention.layers[0].self_attn
            self.assertGreater(float(attention.in_proj_weight.grad.abs().sum()), 0)
        # The time steps of a node now influence each other: changing the end of the signal
        # changes the features that the start contributes.
        encoder = model.node_signal.eval()
        signal = self.batch.signal
        changed = signal.clone()
        changed[:, 400:] = changed[:, 400:] * 3
        pairs = torch.zeros(4, 22, 22, 2)
        with torch.no_grad():
            encoder.out.weight.zero_()
            encoder.out.weight[0, 0] = 1.0                   # read out the first filter at the first time step
            self.assertFalse(torch.allclose(encoder(signal, pairs)[:, 0], encoder(changed, pairs)[:, 0], atol=1e-6))
        for bad in (dict(node_time_attention=1), dict(node_time_attention=1, node_signal="power"),
                    dict(node_time_attention=-1, node_signal="timed")):
            with self.assertRaises(ValueError):
                ModelConfig(**bad)

    def test_low_frequency_path_and_extra_features(self):
        generator = torch.Generator().manual_seed(2)
        graphs = []
        for _ in range(4):
            graph = random_graph(generator)
            graph.x = torch.rand(22, 7, generator=generator) + 0.1
            graph.edge_attr = torch.randn(462, 3, generator=generator)
            graph.signal_low = torch.randn(22, 50, generator=generator)
            graphs.append(graph)
        batch = Batch.from_data_list(graphs)
        torch.manual_seed(0)
        model = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, node_features=7, edge_features=3,
                                          low_samples=50)).eval()
        arguments = (batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.signal)
        # The low-frequency path starts at zero ...
        start = model(*arguments, batch.signal_low)
        torch.testing.assert_close(start, model(*arguments, batch.signal_low * 3))
        # ... and contributes once its output layer has weights.
        torch.nn.init.normal_(model.low.out.weight, std=0.01)
        self.assertFalse(torch.allclose(model(*arguments, batch.signal_low), start, atol=1e-6))
        with self.assertRaises(ValueError):          # the model needs the low signal
            model(*arguments)
        model.train()
        model(*arguments, batch.signal_low).sum().backward()
        self.assertGreater(float(model.low.spatial.grad.abs().sum()), 0)
        # It is added to a selection of paths as well.
        graph_only = EEGGraphModel(ModelConfig(num_electrodes=22, signal_samples=500, node_features=7,
                                               edge_features=3, low_samples=50, paths=("graph",))).eval()
        self.assertEqual(tuple(graph_only(*arguments, batch.signal_low).shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
