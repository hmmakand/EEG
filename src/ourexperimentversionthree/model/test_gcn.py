"""Tests for the manuscript-faithful EEGGCN1 classifier."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.ourexperimentversionthree.model import (
    EEGGCN1,
    EEGGCN1_INPUT_FEATURES,
    EEGGCN1_NODES,
    EEGGCN1Config,
)


def _dense_graph(offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one deterministic 29-node graph as dense (x, adj) tensors."""

    x = (
        torch.arange(EEGGCN1_NODES * EEGGCN1_INPUT_FEATURES, dtype=torch.float32).reshape(
            EEGGCN1_NODES, EEGGCN1_INPUT_FEATURES
        )
        / 100
        + offset
    )
    ring = torch.eye(EEGGCN1_NODES, dtype=torch.float32)
    ring += torch.roll(torch.eye(EEGGCN1_NODES, dtype=torch.float32), shifts=1, dims=0)
    ring += torch.roll(torch.eye(EEGGCN1_NODES, dtype=torch.float32), shifts=-1, dims=0)
    adjacency = (ring > 0).to(torch.float32)
    return x, adjacency


def _batch(*offsets: float) -> tuple[torch.Tensor, torch.Tensor]:
    graphs = [_dense_graph(offset) for offset in offsets]
    x = torch.stack([graph[0] for graph in graphs])
    adjacency = torch.stack([graph[1] for graph in graphs])
    return x, adjacency


class EEGGCN1ConfigTests(unittest.TestCase):
    def test_default_is_dedicated_to_broadcast_11_input(self) -> None:
        config = EEGGCN1Config()
        self.assertEqual(EEGGCN1_INPUT_FEATURES, 11)
        self.assertEqual(EEGGCN1_NODES, 29)
        self.assertEqual(config.input_features, 11)
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
            single_x, single_adj = _batch(0.0)
            single_log_probabilities = self.model(single_x, single_adj)
            batch_x, batch_adj = _batch(0.0, 0.5)
            batch_log_probabilities = self.model(batch_x, batch_adj)
        self.assertEqual(tuple(single_log_probabilities.shape), (1, 2))
        self.assertEqual(tuple(batch_log_probabilities.shape), (2, 2))
        self.assertTrue(torch.isfinite(single_log_probabilities).all())
        self.assertTrue(torch.isfinite(batch_log_probabilities).all())
        # Log-softmax rows must exponentiate to a valid probability distribution.
        probabilities = torch.exp(batch_log_probabilities)
        torch.testing.assert_close(
            probabilities.sum(dim=1), torch.ones(2), rtol=1e-5, atol=1e-5
        )

    def test_adjacency_affects_logits_and_eval_is_deterministic(self) -> None:
        x, adjacency = _batch(0.0)
        # A different, non-isomorphic-under-symmetry adjacency: fully connect
        # the first half of the nodes to each other instead of the ring.
        changed_adjacency = adjacency.clone()
        half = EEGGCN1_NODES // 2
        changed_adjacency[:, :half, :half] = 1.0

        self.model.eval()
        with torch.inference_mode():
            first = self.model(x, adjacency)
            repeated = self.model(x, adjacency)
            changed = self.model(x, changed_adjacency)

        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.allclose(first, changed))

    def test_batch_supports_backward(self) -> None:
        x, adjacency = _batch(0.0, 0.5)
        labels = torch.tensor([0, 1], dtype=torch.long)
        log_probabilities = self.model(x, adjacency)
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
        _, adjacency = _batch(0.0)
        bad_x = torch.zeros(1, EEGGCN1_NODES, 10)
        with self.assertRaisesRegex(ValueError, r"\(batch, 29, 11\)"):
            self.model(bad_x, adjacency)

    def test_invalid_adjacency_shape_is_rejected(self) -> None:
        x, _ = _batch(0.0)
        bad_adjacency = torch.zeros(1, EEGGCN1_NODES, EEGGCN1_NODES - 1)
        with self.assertRaisesRegex(ValueError, r"\(batch, 29, 29\)"):
            self.model(x, bad_adjacency)

    def test_non_finite_inputs_are_rejected(self) -> None:
        x, adjacency = _batch(0.0)
        x[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(x, adjacency)


if __name__ == "__main__":
    unittest.main()
