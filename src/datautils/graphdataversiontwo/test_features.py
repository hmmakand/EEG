"""Numerical checks for feature meaning, amplitude scaling, and graph alignment."""

from __future__ import annotations

import unittest

import numpy as np

from .edge_features import build_edge_features, build_edge_index, compute_connectivity
from .node_features import build_node_features, compute_band_powers, compute_psd, compute_hjorth_features, compute_spectral_entropy


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
        self.assertGreater(features[1, 5] - features[0, 5], 0.5)
        self.assertTrue(np.all((features[:, 5] >= 0) & (features[:, 5] <= 1)))

    def test_hjorth_invariance_and_first_six_unchanged(self):
        signal = np.random.default_rng(7).normal(size=(3, 2000)) * 1e-6
        original = signal.copy()
        result = compute_hjorth_features(signal)
        np.testing.assert_allclose(compute_hjorth_features(signal * 10 + 0.001), result, rtol=1e-12)
        np.testing.assert_array_equal(signal, original)
        freqs, psd = compute_psd(signal, 500)
        expected = np.column_stack((10 * np.log10(np.maximum(compute_band_powers(psd, freqs), 1e-12)),
                                    compute_spectral_entropy(psd, freqs)))
        actual = build_node_features(signal, 500)
        self.assertEqual(actual.shape, (3, 8))
        np.testing.assert_array_equal(actual[:, :6], expected)
        np.testing.assert_array_equal(actual[:, 6:], result)

    def test_hjorth_sinusoid_reference_and_invalid_inputs(self):
        omega = 2 * np.pi * 10 / 500
        signal = np.sin(omega * np.arange(50000))[None, :]
        mobility, complexity = compute_hjorth_features(signal)[0]
        self.assertAlmostEqual(mobility, 2 * np.sin(omega / 2), delta=1e-5)
        self.assertAlmostEqual(complexity, 1, delta=1e-4)
        for bad in (np.ones((2, 20)), np.arange(20)[None, :], np.ones((1, 2)),
                    np.full((1, 20), np.nan), np.full((1, 20), np.inf), np.empty((0, 20))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                compute_hjorth_features(bad)

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
