"""Known phase relationships and configuration switching for PLV and PLI."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import default_dataset_config
from dataset.connectivity import compute_connectivity, compute_pli_matrix, compute_plv_matrix, compute_coherence_matrix
from dataset.validation import validate_config


class ConnectivityTests(unittest.TestCase):
    def test_method_validation_and_default(self):
        config = default_dataset_config()
        self.assertEqual(config.connectivity.method, 'coherence')
        for method in ('plv', 'pli', 'wpli', 'coherence'):
            validate_config(replace(config, connectivity=replace(config.connectivity, method=method)))
        with self.assertRaisesRegex(ValueError, 'method'):
            validate_config(replace(config, connectivity=replace(config.connectivity, method='unknown')))
        for diagonal in (-.1, 1.1, float('nan')):
            with self.assertRaisesRegex(ValueError, 'diagonal'):
                validate_config(replace(config, connectivity=replace(config.connectivity, diagonal=diagonal)))

    def test_constant_lag_and_identical_signals(self):
        time = np.arange(1000) / 250
        wave = np.sin(2 * np.pi * 10 * time)
        signal = np.column_stack([wave, wave, np.sin(2 * np.pi * 10 * time + np.pi / 2)])
        pli = compute_pli_matrix(signal)
        np.testing.assert_allclose(pli, [[0, 0, 1], [0, 0, 1], [1, 1, 0]], atol=1e-12)
        plv = compute_plv_matrix(signal)
        self.assertAlmostEqual(plv[0, 1], 1.)
        for method, expected in [('pli', pli), ('plv', plv)]:
            config = replace(default_dataset_config().connectivity, method=method)
            np.testing.assert_allclose(compute_connectivity(signal, config), expected)
        np.testing.assert_allclose(np.diag(compute_pli_matrix(signal, diagonal=.5)), .5)

    def test_balanced_phase_lags_cancel(self):
        phases = np.column_stack([np.zeros(100), np.tile([np.pi / 2, -np.pi / 2], 50)])
        with patch('dataset.connectivity.extract_phase', return_value=phases):
            matrix = compute_pli_matrix(np.zeros((100, 2)))
        np.testing.assert_array_equal(matrix, np.zeros((2, 2)))

    def test_random_signal_range_and_symmetry(self):
        signal = np.random.default_rng(12).normal(size=(1000, 5))
        matrix = compute_pli_matrix(signal)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(((matrix >= 0) & (matrix <= 1)).all())
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), np.zeros(5))

    def test_coherence_identical_and_independent_signals(self):
        rng = np.random.default_rng(37)
        wave = rng.normal(size=8192)
        signal = np.column_stack([wave, wave, rng.normal(size=8192)])
        matrix = compute_coherence_matrix(signal, 250, 8, 30)
        self.assertAlmostEqual(matrix[0, 1], 1.)
        self.assertLess(matrix[0, 2], .1)
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), [0, 0, 0])
        self.assertTrue(((matrix >= 0) & (matrix <= 1)).all())
        actual = compute_connectivity(signal, default_dataset_config().connectivity, sample_rate=250)
        np.testing.assert_allclose(actual, matrix)

    def test_coherence_band_and_sampling_rate(self):
        from scipy.signal import coherence
        rng = np.random.default_rng(51)
        signal = rng.normal(size=(1000, 2))
        config = replace(default_dataset_config().connectivity, fmin=15, fmax=45, nperseg=128, noverlap=32)
        actual = compute_connectivity(signal, config, sample_rate=500)
        freqs, values = coherence(signal[:, 0], signal[:, 1], fs=500,
                                  nperseg=128, noverlap=32, window='hann', detrend='constant')
        expected = np.mean(values[(freqs >= 15) & (freqs <= 45)])
        self.assertAlmostEqual(actual[0, 1], expected)
        with self.assertRaisesRegex(ValueError, 'sampling rate'):
            compute_connectivity(signal, config)

    def test_coherence_rejects_invalid_settings(self):
        config = default_dataset_config()
        for settings in ({'fmin': 30, 'fmax': 8}, {'fmax': 126}, {'nperseg': 1000},
                         {'nperseg': 0}, {'noverlap': 256}, {'fmin': 8.01, 'fmax': 8.02},
                         {'window': 'unknown_window'}, {'detrend': 'unknown'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_config(replace(config, connectivity=replace(config.connectivity, **settings)))
        # Coherence-only settings do not interfere with another selected method.
        validate_config(replace(config, connectivity=replace(config.connectivity, method='plv', nperseg=0)))

    def test_coherence_rejects_invalid_signals(self):
        with self.assertRaisesRegex(ValueError, 'constant'):
            compute_coherence_matrix(np.zeros((1000, 2)), 250, 8, 30)
        with self.assertRaisesRegex(ValueError, 'finite'):
            compute_coherence_matrix(np.full((1000, 2), np.nan), 250, 8, 30)
        with self.assertRaisesRegex(ValueError, 'two Welch segments'):
            compute_coherence_matrix(np.ones((256, 2)), 250, 8, 30)

    def test_coherence_results_filename(self):
        from train.reporting import resolve_results_path
        self.assertEqual(resolve_results_path(default_dataset_config().connectivity.method).name,
                         'BCI_IV_2a_GAT_COHERENCE_Results.json')


if __name__ == '__main__':
    unittest.main()
