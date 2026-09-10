"""Tests for explicit broadcast-11 feature construction and saved data."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .config import Broadcast11Config
from .features import NODE_FEATURE_NAMES, calculate_broadcast_11_features
from .generate_dataset import generate_broadcast_11_dataset
from .manuscript_features import (
    BetweennessConfig,
    PSDConfig,
    calculate_manuscript_candidate_features,
)
from .topology_features import TopologyFeatureConfig
from .verify_dataset import verify_broadcast_11_dataset


class Broadcast11FeatureTests(unittest.TestCase):
    def test_combines_local_columns_with_exact_graph_value_broadcasts(self) -> None:
        rng = np.random.default_rng(42)
        signals = rng.normal(size=(29, 2000))
        weights = rng.uniform(0.2, 0.9, size=(29, 29))
        plv = ((weights + weights.T) / 2).astype(np.float32)
        np.fill_diagonal(plv, 1.0)
        adjacency = plv.copy()
        np.fill_diagonal(adjacency, 0.0)

        result = calculate_broadcast_11_features(
            signals, plv, adjacency, 500.0
        )
        source = calculate_manuscript_candidate_features(
            signals,
            plv,
            adjacency,
            500.0,
            psd_config=PSDConfig(
                reducer="mean_density", frequency_band_hz=(31.0, 40.0)
            ),
            betweenness_config=BetweennessConfig(
                graph_mode="threshold_binary", threshold=0.3
            ),
            topology_config=TopologyFeatureConfig(),
        )
        self.assertEqual(result.node_features.shape, (29, 11))
        self.assertEqual(len(NODE_FEATURE_NAMES), 11)
        np.testing.assert_array_equal(result.node_features[:, :4], source.node_features)
        np.testing.assert_array_equal(
            result.node_features[:, 4:],
            np.broadcast_to(source.graph_features, (29, 7)),
        )

    def test_small_dataset_is_generated_atomically_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "broadcast11"
            generated = generate_broadcast_11_dataset(
                Broadcast11Config(output_dir=output_dir), limit=2
            )
            self.assertEqual(generated, output_dir.resolve())
            verify_broadcast_11_dataset(generated)
            node_features = np.load(generated / "node_features.npy")
            self.assertEqual(node_features.shape, (2, 29, 11))
            with self.assertRaises(FileExistsError):
                generate_broadcast_11_dataset(
                    Broadcast11Config(output_dir=output_dir), limit=1
                )


if __name__ == "__main__":
    unittest.main()
