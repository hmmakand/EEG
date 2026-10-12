"""Known phase relationships and PLV-only configuration."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import default_dataset_config
from dataset.connectivity import compute_connectivity, compute_plv_matrix
from dataset.validation import validate_config


class ConnectivityTests(unittest.TestCase):
    def test_method_validation_and_default(self):
        config = default_dataset_config()
        self.assertEqual(config.connectivity.method, 'plv')
        validate_config(config)
        for method in ('ciplv', 'pli', 'wpli', 'coherence', 'consensus', 'rplv', 'unknown'):
            with self.subTest(method=method):
                connectivity = replace(config.connectivity, method=method)
                with self.assertRaisesRegex(ValueError, 'method must be plv'):
                    validate_config(replace(config, connectivity=connectivity))
                with self.assertRaisesRegex(ValueError, 'method must be plv'):
                    compute_connectivity(np.ones((100, 2)), connectivity)
        for diagonal in (-.1, 1.1, float('nan')):
            with self.assertRaisesRegex(ValueError, 'diagonal'):
                validate_config(replace(config, connectivity=replace(config.connectivity, diagonal=diagonal)))

    def test_constant_lag_and_identical_signals(self):
        time = np.arange(1000) / 250
        wave = np.sin(2 * np.pi * 10 * time)
        signal = np.column_stack([wave, wave, np.sin(2 * np.pi * 10 * time + np.pi / 2)])
        plv = compute_plv_matrix(signal)
        np.testing.assert_allclose(plv, np.ones((3, 3)) - np.eye(3), atol=1e-12)
        config = replace(default_dataset_config().connectivity, diagonal=.5)
        expected = plv.copy()
        np.fill_diagonal(expected, .5)
        np.testing.assert_allclose(compute_connectivity(signal, config), expected)

    def test_balanced_phase_lags_cancel(self):
        phases = np.column_stack([np.zeros(100), np.tile([np.pi / 2, -np.pi / 2], 50)])
        with patch('dataset.connectivity.extract_phase', return_value=phases):
            matrix = compute_plv_matrix(np.zeros((100, 2)))
        np.testing.assert_allclose(matrix, np.zeros((2, 2)), atol=1e-12)

    def test_random_signal_range_and_symmetry(self):
        signal = np.random.default_rng(12).normal(size=(1000, 5))
        matrix = compute_plv_matrix(signal)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(((matrix >= 0) & (matrix <= 1)).all())
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), np.zeros(5))

    def test_plv_results_filename(self):
        from train.reporting import resolve_results_path
        self.assertEqual(resolve_results_path(default_dataset_config().connectivity.method, threshold=0.35).name,
                         'BCI_IV_2a_GAT_PLV_Threshold_0.35_Results.json')


if __name__ == '__main__':
    unittest.main()
