"""Numerical checks for feature meaning, amplitude scaling, and graph alignment."""

from __future__ import annotations

import unittest

import numpy as np

from .edge_features import build_edge_features, build_edge_index, compute_connectivity
from .node_features import build_node_features, compute_band_powers, compute_psd


class FeatureTests(unittest.TestCase):
    """Check physical and graph invariants independently of implementation details."""

    def test_amplitude_scaling_changes_power_but_preserves_entropy(self) -> None:
        """Tenfold signal amplitude adds 20 dB without changing spectral entropy."""
        signal = np.random.default_rng(9).normal(scale=1e-6, size=(3, 2000))
        original = signal.copy()
        features = build_node_features(signal, 500.0)
        scaled = build_node_features(10 * signal, 500.0)
        np.testing.assert_allclose(scaled[:, :5] - features[:, :5], 20.0, atol=1e-12)
        np.testing.assert_allclose(scaled[:, 5], features[:, 5], atol=1e-14)
        np.testing.assert_array_equal(signal, original)

    def test_alpha_tone_has_concentrated_alpha_power_and_lower_entropy(self) -> None:
        """A 10-Hz tone concentrates power in alpha and has less entropy than noise."""
        times = np.arange(2000) / 500.0
        tone = 1e-6 * np.sin(2 * np.pi * 10 * times)
        noise = np.random.default_rng(12).normal(scale=1e-6, size=2000)
        signal = np.stack([tone, noise])
        frequencies, psd = compute_psd(signal, 500.0)
        powers = compute_band_powers(psd, frequencies)
        features = build_node_features(signal, 500.0)
        self.assertEqual(int(np.argmax(powers[0])), 2)
        self.assertGreater(powers[0, 2] / powers[0].sum(), 0.999)
        self.assertGreater(features[1, -1] - features[0, -1], 0.5)
        self.assertTrue(np.all((features[:, -1] >= 0) & (features[:, -1] <= 1)))

    def test_phase_locked_channels_and_reverse_edges(self) -> None:
        """Quadrature alpha signals produce strong connectivity on both edge directions."""
        times = np.arange(2000) / 500.0
        signal = np.stack([
            np.sin(2 * np.pi * 10 * times),
            np.sin(2 * np.pi * 10 * times + np.pi / 2),
            np.random.default_rng(24).normal(size=2000),
        ]) * 1e-6
        connectivity = compute_connectivity(signal, 500.0)
        edge_index = build_edge_index(3)
        edge_features = build_edge_features(connectivity, edge_index)
        self.assertEqual(edge_index.shape, (2, 6))
        self.assertFalse(np.any(edge_index[0] == edge_index[1]))
        for name, adjacency in connectivity.items():
            with self.subTest(method=name):
                # Band/time smoothing reduces iCoh magnitude even for pure tones.
                # The locked pair must still exceed either noise pairing.
                self.assertGreater(adjacency[2, 0, 1], 0.8)
                self.assertGreater(adjacency[2, 0, 1] - max(adjacency[2, 0, 2], adjacency[2, 1, 2]), 0.5)
                np.testing.assert_array_equal(edge_features[name][:3], edge_features[name][3:])
                np.testing.assert_array_equal(adjacency, adjacency.transpose(0, 2, 1))
                self.assertTrue(np.isfinite(edge_features[name]).all())
        self.assertLess(connectivity['plv'][2, 0, 2], 0.6)


if __name__ == "__main__":
    unittest.main()
