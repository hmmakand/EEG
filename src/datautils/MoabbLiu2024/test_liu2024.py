"""Tests for the Liu2024 preparation package.

Run from the repository root with:
    .venv/bin/python -m unittest src.datautils.MoabbLiu2024.test_liu2024
"""

import unittest

import numpy as np

from .channels import select_eeg_channels
from .config import DEFAULT_DATA_ROOT, Liu2024Config
from .loader import load_liu2024
from .preprocessing import preprocess_liu2024
from .validation import validate_liu2024
from .windowing import create_left_right_windows


class Liu2024ConfigurationTests(unittest.TestCase):
    def test_default_label_mapping(self):
        self.assertEqual(
            Liu2024Config(subject_ids=(1,)).event_mapping,
            {"left_hand": 0, "right_hand": 1},
        )

    def test_invalid_subject_is_rejected(self):
        with self.assertRaises(ValueError):
            Liu2024Config(subject_ids=(0,))

    def test_standardization_block_tracks_sampling_frequency(self):
        config = Liu2024Config(subject_ids=(1,))
        self.assertEqual(config.standardize_init_block_seconds, 4.0)
        self.assertEqual(
            config.standardize_init_block_size,
            round(4.0 * config.target_sfreq),
        )


@unittest.skipUnless(
    (DEFAULT_DATA_ROOT / "MNE-liu2024-data" / "files" / "edffile").is_dir(),
    "Local Liu2024 data is unavailable",
)
class Liu2024LocalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_liu2024([1])

    def test_loader_and_raw_validation(self):
        summary = validate_liu2024(self.dataset)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["subject"], 1)
        self.assertEqual(summary[0]["left_hand_trials"], 20)
        self.assertEqual(summary[0]["right_hand_trials"], 20)

    def test_complete_preparation_stages(self):
        # Reload because Braindecode preprocessing intentionally mutates datasets.
        dataset = load_liu2024([1])
        select_eeg_channels(dataset)
        eeg_summary = validate_liu2024(dataset, require_eeg_only=True)
        self.assertEqual(eeg_summary[0]["n_eeg_channels"], 29)

        config = Liu2024Config(subject_ids=(1,))
        preprocess_liu2024(
            dataset,
            l_freq=config.l_freq,
            h_freq=config.h_freq,
            target_sfreq=config.target_sfreq,
            standardize=config.standardize,
            factor_new=config.standardize_factor_new,
            init_block_size=config.standardize_init_block_size,
        )
        validate_liu2024(
            dataset,
            require_eeg_only=True,
            expected_sfreq=config.target_sfreq,
        )
        windows = create_left_right_windows(
            dataset,
            event_mapping=config.event_mapping,
            trial_start_offset_seconds=config.trial_start_offset_seconds,
            trial_stop_offset_seconds=config.trial_stop_offset_seconds,
        )

        self.assertEqual(len(windows), 40)
        x, y, window_info = windows[0]
        self.assertEqual(x.shape, (29, round(4.0 * config.target_sfreq)))
        self.assertIn(int(y), (0, 1))
        self.assertEqual(len(window_info), 3)
        self.assertTrue(np.isfinite(x).all())
        targets = [int(windows[index][1]) for index in range(len(windows))]
        self.assertEqual(targets.count(0), 20)
        self.assertEqual(targets.count(1), 20)


if __name__ == "__main__":
    unittest.main()
