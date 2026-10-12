"""The synthetic-signal checks separate faithful from unfaithful generated trials."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasetsynthetic.evaluate import compare, summarize


def trials(count, rng, cue_signal=True, tone=0.0):
    """[count, 1000, 8] noise; class 1 gets extra 12 Hz power on channel 0 after the cue."""
    labels = np.repeat([0, 1], count // 2)
    signals = rng.normal(size=(count, 1000, 8))
    time = np.arange(1000) / 250
    wave = np.sin(2 * np.pi * 12 * time)
    start = 500 if cue_signal else 0
    signals[labels == 1, start:, 0] += 2 * wave[start:]
    signals += tone * np.sin(2 * np.pi * 20 * time)[None, :, None]
    return signals, labels


class SyntheticEvaluationTests(unittest.TestCase):
    def test_faithful_and_unfaithful_generators(self):
        rng = np.random.default_rng(0)
        real, real_labels = trials(80, rng)
        settings = dict(sample_rate=250, edges=(8, 30), window=(2.0, 4.0))

        faithful = compare(real, real_labels, *trials(60, rng), **settings)
        self.assertGreater(faithful['real_to_real'], .9)
        self.assertGreater(faithful['real_to_synthetic'], .9)
        self.assertGreater(faithful['synthetic_to_real'], .9)
        self.assertLess(faithful['pre_cue_synthetic_to_synthetic'], .7)
        self.assertLess(faithful['real_versus_synthetic'], .7)
        np.testing.assert_allclose(faithful['band_power_ratio'], 1, atol=.15)
        self.assertAlmostEqual(faithful['majority_guess'], 80 / 140)

        # Class signal through the whole trial and an extra 20 Hz tone.
        unfaithful = compare(real, real_labels, *trials(60, rng, cue_signal=False, tone=1.5), **settings)
        self.assertGreater(unfaithful['pre_cue_synthetic_to_synthetic'], .9)
        self.assertGreater(unfaithful['real_versus_synthetic'], .95)
        self.assertGreater(unfaithful['band_power_ratio'][3], 2)  # 13-30 Hz

        summary = summarize([faithful, unfaithful])
        self.assertAlmostEqual(summary['real_to_real'], (faithful['real_to_real'] + unfaithful['real_to_real']) / 2)
        self.assertEqual(len(summary['band_power_ratio']), 5)


if __name__ == '__main__':
    unittest.main()
