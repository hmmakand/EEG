"""The Liu 2024 reader: labels, epochs, and the imagery onset of each trial."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..data import bci2a_config
from ..data import LIU2024_DUPLICATES, build_subject_graphs, liu2024_config, read_trials, subject_file
from ..data.liu2024 import (EEG_CHANNELS, cut_trials, mirror_order, mirror_trial, read_labels, read_paralysed_sides,
                            read_recording)
from ..data.signal import analysed_signal

CONFIG = liu2024_config()
# Set LIU2024_DATA to another location to run the checks on the real recordings there.
REAL_DATA = Path(os.environ.get("LIU2024_DATA", "/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data"))


def recording(onsets, rng):
    """A recording of len(onsets) epochs at 500 Hz with markers 1 (epoch), 2 (imagery) and 3 (end)."""
    eeg = rng.normal(size=(4000 * len(onsets), 29))
    marker = np.zeros(len(eeg), dtype=int)
    for trial, onset in enumerate(onsets):
        start = 4000 * trial
        marker[start] = 1
        marker[start + onset] = 2
        marker[start + onset + 2000] = 3
    return eeg, marker


class Liu2024Tests(unittest.TestCase):
    def test_paths_accept_the_moabb_folder_or_its_files_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "files").mkdir()
            expected = Path(folder) / "files" / "edffile" / "sub-07" / "eeg" / "sub-07_task-motor-imagery_eeg.edf"
            self.assertEqual(subject_file(folder, 7, CONFIG), expected)
            self.assertEqual(subject_file(Path(folder) / "files", 7, CONFIG), expected)

    def test_one_label_per_trial_from_the_events_table(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                read_labels(folder)
            rows = ["onset\tduration\ttrial_type\tresponse_time\tvalue\tstim_file"]
            for trial, trial_type in enumerate((1, 2, 2, 1)):
                rows += [f"{trial * 8000 + onset}\t{length}\t{trial_type}\tN/A\t{stage}\tstim"
                         for onset, length, stage in ((1, 2000, 1), (2001, 4000, 2), (6001, 2000, 3))]
            (Path(folder) / "38516084").write_text("\r\n".join(rows) + "\r\n")
            self.assertEqual(read_labels(folder), [0, 1, 1, 0])

    def test_epochs_are_halved_in_rate_and_each_trial_keeps_its_own_onset(self):
        eeg, marker = recording([1004, 1, 1830], np.random.default_rng(0))
        marker[4000 + 3004] = 2                      # a stray marker late in the second epoch
        trials = cut_trials(eeg, marker, [0, 1, 0], 5, CONFIG)
        self.assertEqual([(t.onset, t.label, t.subject_id, t.session_id, t.trial_id) for t in trials],
                         [(502, 0, 5, 0, 0), (0, 1, 5, 0, 1), (915, 0, 5, 0, 2)])
        for number, trial in enumerate(trials):
            np.testing.assert_array_equal(trial.signal, eeg[4000 * number:4000 * (number + 1):2])
        # The analysed window is the 4 s that follow the trial's own onset.
        self.assertEqual([analysed_signal(t, CONFIG).shape for t in trials], [(1000, 29)] * 3)

    def test_mirroring_exchanges_the_hemispheres_and_the_hands(self):
        order = mirror_order()
        self.assertEqual(sorted(order), list(range(29)))
        self.assertEqual([order[order[i]] for i in range(29)], list(range(29)))      # mirroring twice changes nothing
        self.assertEqual(order[EEG_CHANNELS.index("C3")], EEG_CHANNELS.index("C4"))
        midline = [name for i, name in enumerate(EEG_CHANNELS) if order[i] == i]
        self.assertEqual(midline, ["Fz", "FCz", "Cz", "Pz", "Oz"])
        eeg, marker = recording([1004], np.random.default_rng(0))
        trial = cut_trials(eeg, marker, [0], 5, CONFIG)[0]
        mirrored = mirror_trial(trial)
        self.assertEqual((mirrored.label, mirrored.onset), (1, trial.onset))
        np.testing.assert_array_equal(mirrored.signal[:, EEG_CHANNELS.index("C3")],
                                      trial.signal[:, EEG_CHANNELS.index("C4")])
        np.testing.assert_array_equal(mirror_trial(mirrored).signal, trial.signal)
        with self.assertRaises(ValueError):          # the option belongs to liu2024
            bci2a_config(align_paralysed_side=True)

    def test_paralysed_sides_come_from_the_participants_table(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                read_paralysed_sides(folder)
            (Path(folder) / "participants.tsv").write_text(
                "Participant_ID\tGender\tParalysisSide\nsub-01\tmale\tright\nsub-12\tfemale\tleft\n")
            self.assertEqual(read_paralysed_sides(folder), {1: "right", 12: "left"})

    def test_malformed_recordings_are_rejected(self):
        eeg, marker = recording([1004, 1004], np.random.default_rng(0))
        with self.assertRaises(ValueError):          # a label too many for the recording
            cut_trials(eeg, marker, [0, 1, 0], 1, CONFIG)
        without_imagery = marker.copy()
        without_imagery[4000 + 1004] = 0
        with self.assertRaises(ValueError):
            cut_trials(eeg, without_imagery, [0, 1], 1, CONFIG)
        without_start = marker.copy()
        without_start[4000] = 0
        with self.assertRaises(ValueError):
            cut_trials(eeg, without_start, [0, 1], 1, CONFIG)
        with self.assertRaises(ValueError):          # 300 Hz does not divide 500 Hz
            cut_trials(eeg, marker, [0, 1], 1, liu2024_config(sample_rate=300))

    @unittest.skipUnless(subject_file(REAL_DATA, 1, CONFIG).is_file(), "Liu 2024 recordings are not on this machine")
    def test_real_recording(self):
        self.assertEqual(len(EEG_CHANNELS), 29)
        trials = read_trials(REAL_DATA, 6, CONFIG)
        self.assertEqual(len(trials), 40)
        self.assertEqual([t.label for t in trials], [0, 1] * 20)          # the classes alternate
        self.assertTrue(all(t.signal.shape == (2000, 29) for t in trials))
        self.assertEqual([t.onset for t in trials[:3]], [502, 678, 723])  # this subject's imagery starts late
        graphs = build_subject_graphs(REAL_DATA, 6, CONFIG)
        graph = graphs[0]
        self.assertEqual((tuple(graph.x.shape), tuple(graph.edge_index.shape), tuple(graph.edge_attr.shape),
                          tuple(graph.signal.shape)), ((29, 6), (2, 812), (812, 2), (29, 1000)))
        mean_covariance = np.mean([(g.signal.double() @ g.signal.double().T / 1000).numpy() for g in graphs], axis=0)
        np.testing.assert_allclose(mean_covariance, np.eye(29), atol=1e-4)   # alignment

    @unittest.skipUnless(subject_file(REAL_DATA, 1, CONFIG).is_file(), "Liu 2024 recordings are not on this machine")
    def test_the_excluded_recordings_are_copies(self):
        for copy, original in LIU2024_DUPLICATES.items():
            self.assertNotIn(copy, CONFIG.subjects)
            self.assertIn(original, CONFIG.subjects)
            a, _ = read_recording(subject_file(REAL_DATA, copy, CONFIG))
            b, _ = read_recording(subject_file(REAL_DATA, original, CONFIG))
            self.assertGreater(np.corrcoef(a[:, 0], b[:, 0])[0, 1], 0.9999)
        a, _ = read_recording(subject_file(REAL_DATA, 1, CONFIG))
        b, _ = read_recording(subject_file(REAL_DATA, 2, CONFIG))
        self.assertLess(abs(np.corrcoef(a[:, 0], b[:, 0])[0, 1]), 0.5)

    @unittest.skipUnless(subject_file(REAL_DATA, 1, CONFIG).is_file(), "Liu 2024 recordings are not on this machine")
    def test_alignment_by_paralysed_side_on_real_recordings(self):
        sides = read_paralysed_sides(REAL_DATA)
        self.assertEqual((len(sides), sides[1], sides[2]), (50, "right", "left"))
        aligned = liu2024_config(align_paralysed_side=True)
        plain, mirrored = read_trials(REAL_DATA, 1, CONFIG), read_trials(REAL_DATA, 1, aligned)   # paralysed on the right
        self.assertEqual([t.label for t in mirrored], [1 - t.label for t in plain])
        np.testing.assert_array_equal(mirrored[0].signal[:, 13], plain[0].signal[:, 14])
        untouched = read_trials(REAL_DATA, 2, aligned)                                           # paralysed on the left
        np.testing.assert_array_equal(untouched[0].signal, read_trials(REAL_DATA, 2, CONFIG)[0].signal)


if __name__ == "__main__":
    unittest.main()
