"""Tests for the sparse, edge-weighted EEGGCN1 classifier."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PygDataLoader

from src.ourexperimentversionfour.model import (
    EEGGCN1,
    EEGGCN1_INPUT_FEATURES,
    EEGGCN1_NODES,
    EEGGCN1Config,
)


def _ring_edge_index(nodes: int) -> torch.Tensor:
    """Build a directed ring (each node connected to its two neighbors)."""

    sources = torch.cat([torch.arange(nodes), torch.arange(nodes)])
    targets = torch.cat(
        [torch.roll(torch.arange(nodes), shifts=-1), torch.roll(torch.arange(nodes), shifts=1)]
    )
    return torch.stack([sources, targets])


def _graph(offset: float = 0.0, edge_weight_value: float = 1.0) -> Data:
    """Build one deterministic 29-node sparse graph."""

    x = (
        torch.arange(EEGGCN1_NODES * EEGGCN1_INPUT_FEATURES, dtype=torch.float32).reshape(
            EEGGCN1_NODES, EEGGCN1_INPUT_FEATURES
        )
        / 100
        + offset
    )
    edge_index = _ring_edge_index(EEGGCN1_NODES)
    edge_weight = torch.full((edge_index.shape[1],), edge_weight_value, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))


def _batch(*offsets: float) -> Data:
    loader = PygDataLoader([_graph(offset) for offset in offsets], batch_size=len(offsets))
    return next(iter(loader))


class EEGGCN1ConfigTests(unittest.TestCase):
    def test_default_is_dedicated_to_the_without_csd_alpha_wpli_combination(self) -> None:
        config = EEGGCN1Config()
        self.assertEqual(EEGGCN1_INPUT_FEATURES, 6)
        self.assertEqual(EEGGCN1_NODES, 29)
        self.assertEqual(config.input_features, 6)
        self.assertEqual(config.nodes, 29)

    def test_invalid_configuration_is_rejected(self) -> None:
        for keyword_arguments in (
            {"nodes": 0},
            {"input_features": 0},
            {"gcn_width": -1},
            {"output_classes": 0},
            {"dropout": -0.1},
            {"dropout": 1.0},
            {"leaky_relu_slope": -0.01},
        ):
            with self.subTest(keyword_arguments=keyword_arguments):
                with self.assertRaises(ValueError):
                    EEGGCN1Config(**keyword_arguments)  # type: ignore[arg-type]


class EEGGCN1Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = EEGGCN1(EEGGCN1Config(dropout=0.0))

    def test_single_graph_and_batch_shapes(self) -> None:
        self.model.eval()
        with torch.inference_mode():
            single = _batch(0.0)
            single_log_probabilities = self.model(
                single.x, single.edge_index, single.edge_attr.squeeze(-1), single.batch
            )
            batch = _batch(0.0, 0.5)
            batch_log_probabilities = self.model(
                batch.x, batch.edge_index, batch.edge_attr.squeeze(-1), batch.batch
            )
        self.assertEqual(tuple(single_log_probabilities.shape), (1, 2))
        self.assertEqual(tuple(batch_log_probabilities.shape), (2, 2))
        self.assertTrue(torch.isfinite(single_log_probabilities).all())
        self.assertTrue(torch.isfinite(batch_log_probabilities).all())
        # Log-softmax rows must exponentiate to a valid probability distribution.
        probabilities = torch.exp(batch_log_probabilities)
        torch.testing.assert_close(
            probabilities.sum(dim=1), torch.ones(2), rtol=1e-5, atol=1e-5
        )

    def test_no_batch_argument_assumes_one_graph(self) -> None:
        self.model.eval()
        graph = _graph(0.0)
        with torch.inference_mode():
            with_batch = self.model(
                graph.x,
                graph.edge_index,
                graph.edge_attr.squeeze(-1),
                torch.zeros(EEGGCN1_NODES, dtype=torch.long),
            )
            without_batch = self.model(graph.x, graph.edge_index, graph.edge_attr.squeeze(-1))
        self.assertTrue(torch.equal(with_batch, without_batch))

    def test_edge_weight_affects_logits_and_eval_is_deterministic(self) -> None:
        graph = _graph(0.0, edge_weight_value=1.0)
        changed = _graph(0.0, edge_weight_value=0.1)

        self.model.eval()
        with torch.inference_mode():
            first = self.model(graph.x, graph.edge_index, graph.edge_attr.squeeze(-1))
            repeated = self.model(graph.x, graph.edge_index, graph.edge_attr.squeeze(-1))
            different = self.model(changed.x, changed.edge_index, changed.edge_attr.squeeze(-1))

        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.allclose(first, different))

    def test_batch_supports_backward(self) -> None:
        batch = _batch(0.0, 0.5)
        labels = torch.tensor([0, 1], dtype=torch.long)
        log_probabilities = self.model(
            batch.x, batch.edge_index, batch.edge_attr.squeeze(-1), batch.batch
        )
        # The model outputs log-softmax already, so NLLLoss is the correct
        # pairing -- CrossEntropyLoss would incorrectly apply softmax again.
        loss = nn.NLLLoss()(log_probabilities, labels)
        loss.backward()

        self.assertEqual(tuple(log_probabilities.shape), (2, 2))
        self.assertTrue(torch.isfinite(loss))
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(
                torch.isfinite(gradient).all()
                for gradient in gradients
                if gradient is not None
            )
        )

    def test_invalid_node_feature_shape_is_rejected(self) -> None:
        graph = _graph(0.0)
        bad_x = torch.zeros(EEGGCN1_NODES, 10)
        with self.assertRaisesRegex(ValueError, r"total_nodes, 6"):
            self.model(bad_x, graph.edge_index, graph.edge_attr.squeeze(-1))

    def test_mismatched_edge_weight_shape_is_rejected(self) -> None:
        graph = _graph(0.0)
        bad_edge_weight = torch.zeros(graph.edge_index.shape[1] - 1)
        with self.assertRaisesRegex(ValueError, "edge_weight must have shape"):
            self.model(graph.x, graph.edge_index, bad_edge_weight)

    def test_non_finite_inputs_are_rejected(self) -> None:
        graph = _graph(0.0)
        graph.x[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(graph.x, graph.edge_index, graph.edge_attr.squeeze(-1))


if __name__ == "__main__":
    unittest.main()
