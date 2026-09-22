"""Regression checks for real source marker anomalies and CSD signal isolation."""

import unittest

import mne
import numpy as np
from mne.preprocessing import compute_current_source_density
from threadpoolctl import threadpool_limits

from .config import SOURCE_ROOT
from .preprocessing import attach_analysis_montage, compute_csd, prepare_eeg
from .source import discover_recordings, load_recording, read_trial_table


@unittest.skipUnless((SOURCE_ROOT / 'files/edffile').is_dir(), 'Local Liu2024 data is unavailable')
class SourceTests(unittest.TestCase):
    """Exercise recorded marker exceptions using the local source files."""

    def test_extra_mi_pulses_do_not_shift_subject43_trials(self):
        """Two extra MI pulses after breaks must not become graph samples."""
        path = discover_recordings(SOURCE_ROOT)[43]
        raw = load_recording(path)
        table = read_trial_table(raw, 43, path)
        self.assertEqual(len(table), 40)
        self.assertEqual(table.target.value_counts().to_dict(), {'left_hand': 20, 'right_hand': 20})
        self.assertNotIn(99005, table.start_sample.tolist())
        self.assertNotIn(103003, table.start_sample.tolist())
        self.assertEqual(table.loc[25, 'start_sample'], 100001)
        self.assertEqual(table.loc[26, 'start_sample'], 104001)
        self.assertEqual(table.loc[25, 'target'], 'right_hand')
        self.assertEqual(table.loc[26, 'target'], 'left_hand')
        raw.close()

    def test_fractional_tail_is_not_an_event(self):
        """Subject 13's fractional EDF tail must not create a 41st imagery trial."""
        path = discover_recordings(SOURCE_ROOT)[13]
        raw = load_recording(path)
        table = read_trial_table(raw, 13, path)
        self.assertEqual(len(table), 40)
        self.assertGreater(table.attrs['marker_audit']['nonintegral_marker_sample_count'], 0)
        self.assertTrue((table.stop_sample <= 159769).all())
        raw.close()

    def test_csd_matches_mne_auto_and_preserves_eeg(self):
        """Explicit recorded sphere reproduces MNE auto CSD without changing EEG."""
        with threadpool_limits(limits=1):
            raw = load_recording(discover_recordings(SOURCE_ROOT)[1])
            eeg = attach_analysis_montage(prepare_eeg(raw))
            before = eeg.get_data(start=0, stop=2000)
            csd, sphere = compute_csd(eeg)
            direct = compute_current_source_density(eeg.copy().load_data(verbose='ERROR'),
                                                    sphere='auto', copy=False, verbose='ERROR')
            assert isinstance(direct, mne.io.BaseRaw)
            np.testing.assert_allclose(csd.get_data(start=0, stop=2000),
                                       direct.get_data(start=0, stop=2000), rtol=0, atol=0)
            np.testing.assert_array_equal(eeg.get_data(start=0, stop=2000), before)
            self.assertFalse(eeg.preload)
            self.assertEqual(set(csd.get_channel_types()), {'csd'})
            self.assertGreater(sphere[3], 0)
            raw.close()
            eeg.close()
            csd.close()
            direct.close()
