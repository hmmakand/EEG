"""Verify actual band boundaries independently of the installed averaging helper."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .config import BANDS
from .edge_features import compute_connectivity


class BandTests(unittest.TestCase):
    """Check requested bands independently of the installed averaging helper."""

    def test_explicit_band_bounds(self):
        """A known frequency ramp must average exactly each named inclusive band."""
        frequencies = np.arange(1., 41.)
        dense = np.zeros((1, 2, 2, 40))
        dense[0, 1, 0] = frequencies / 40
        result = SimpleNamespace(get_data=lambda **kwargs: dense)
        signal = np.random.default_rng(1).normal(size=(2, 2000))
        with patch.object(sys.modules[compute_connectivity.__module__], 'spectral_connectivity_time',
                   return_value=[result, result, result]) as calculate:
            actual = compute_connectivity(signal, 500)
        self.assertFalse(calculate.call_args.kwargs['faverage'])
        expected = np.array([(low + high) / 80 for low, high in BANDS])
        for values in actual.values():
            np.testing.assert_allclose(values[:, 0, 1], expected, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
