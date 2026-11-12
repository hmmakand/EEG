"""Recording -> samples -> aligned samples -> graphs."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ..datascripts import (BCI2A_SUBJECTS, LIU2024_ALL_SUBJECTS, LIU2024_DUPLICATES, LIU2024_SUBJECTS, DataConfig,
                           align_subject, bci2a_config, build_graph, build_subject_graphs, dataset_config,
                           liu2024_config, read_samples, subject_file)
from ..datascripts.liu2024 import EEG_CHANNELS, cut_samples, read_recording
from ..model import ModelConfig
from ..datascripts.roi import roi_features
from ..datascripts.signal import analysed_signal
from .fixture import BCI2A_CONFIG, CONFIG, fake_recordings, recording, write_bci2a_recordings

# Set LIU2024_DATA to another location to run the checks on the real recordings there.
REAL_DATA = Path(os.environ.get("LIU2024_DATA", "/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data"))
HAVE_REAL_DATA = subject_file(REAL_DATA, 1, CONFIG).is_file()


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = cut_samples(*recording(), 5, CONFIG)

    def test_default_rules(self):
        config = liu2024_config()
        self.assertEqual(config, DataConfig())
        self.assertEqual((len(config.subjects), set(LIU2024_ALL_SUBJECTS) - set(config.subjects)), (46, {5, 6, 15, 49}))
        self.assertEqual(config.subjects, LIU2024_SUBJECTS)
        self.assertEqual((config.num_channels, config.trial_samples, config.window_samples), (29, 2000, 475))
        self.assertEqual((config.class_names, config.task), (("rest", "imagery"), "imagery (1) against rest (0)"))
        bci = bci2a_config()
        self.assertEqual((bci.subjects, bci.num_channels, bci.trial_samples, bci.window_samples),
                         (BCI2A_SUBJECTS, 22, 1000, 500))
        self.assertEqual(bci.task, "right hand (1) against left hand (0)")
        self.assertEqual(dataset_config("bci2a"), bci)
        self.assertEqual([(m.num_electrodes, m.signal_samples) for m in map(ModelConfig.for_data, (config, bci))],
                         [(29, 475), (22, 500)])
        with self.assertRaises(ValueError):
            dataset_config("other")
        self.assertEqual(dataset_config("jia2019").num_channels, 63)
        self.assertEqual(config.feature_bands, ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)))
        self.assertEqual(len(EEG_CHANNELS), 29)
        for bad in ((), (1, 1)):
            with self.assertRaises(ValueError):
                DataConfig(subjects=bad)

    def test_paths_accept_the_moabb_folder_or_its_files_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "files").mkdir()
            expected = Path(folder) / "files" / "edffile" / "sub-07" / "eeg" / "sub-07_task-motor-imagery_eeg.edf"
            self.assertEqual(subject_file(folder, 7, CONFIG), expected)
            self.assertEqual(subject_file(Path(folder) / "files", 7, CONFIG), expected)
            self.assertEqual(subject_file(folder, 7, BCI2A_CONFIG), Path(folder) / "A07T.mat")

    def test_two_samples_per_trial_placed_by_the_trials_own_onset(self):
        # The fourth trial's imagery starts too late for its break to fit: it gives no sample.
        self.assertEqual([(s.trial_id, s.sample_id, s.label, s.start) for s in self.samples],
                         [(0, 0, 1, 502 + 250), (0, 1, 0, 502 + 1000), (1, 2, 1, 250), (1, 3, 0, 1000),
                          (2, 4, 1, 505 + 250), (2, 5, 0, 505 + 1000), (4, 8, 1, 503 + 250), (4, 9, 0, 503 + 1000)])
        self.assertEqual(sum(s.label for s in self.samples), 4)                 # balanced
        eeg, _ = recording()
        for sample in self.samples:
            self.assertEqual((sample.signal.shape, sample.subject_id), ((2000, 29), 5))
            np.testing.assert_array_equal(sample.signal, eeg[4000 * sample.trial_id:4000 * (sample.trial_id + 1):2])
        self.assertEqual({analysed_signal(s, CONFIG).shape for s in self.samples}, {(475, 29)})
        # A stray imagery marker late in an epoch is ignored: the first one is the onset.
        eeg, marker = recording()
        marker[3004] = 2
        self.assertEqual([s.start for s in cut_samples(eeg, marker, 5, CONFIG)], [s.start for s in self.samples])

    def test_bci2a_trials(self):
        with tempfile.TemporaryDirectory() as folder:
            write_bci2a_recordings(folder)
            samples = read_samples(folder, 1, BCI2A_CONFIG)
            # Sessions 1 and 2; left/right only; feet and the cut-off trial are skipped.
            self.assertEqual([s.label for s in samples], [0, 1, 1, 0] * 2)
            self.assertEqual([(s.trial_id, s.sample_id) for s in samples], [(i, i) for i in range(8)])
            self.assertTrue(all(s.signal.shape == (1000, 22) and s.start == 500 and s.subject_id == 1 for s in samples))
            with self.assertRaises(ValueError):
                read_samples(folder, 1, bci2a_config(subjects=(1,), session_indices=(7,)))
            # Without explicit indices every session that contains trials is read.
            self.assertEqual(len(read_samples(folder, 1, bci2a_config(subjects=(1,)))), 12)
            graphs = build_subject_graphs(folder, 1, BCI2A_CONFIG)
            self.assertEqual((tuple(graphs[0].x.shape), tuple(graphs[0].edge_attr.shape), tuple(graphs[0].signal.shape),
                              tuple(graphs[0].roi.shape)), ((22, 6), (462, 2), (22, 500), (1, 4)))
            covariance = np.mean([(g.signal.double() @ g.signal.double().T / 500).numpy() for g in graphs], axis=0)
            np.testing.assert_allclose(covariance, np.eye(22), atol=1e-4)

    def test_malformed_recordings_are_rejected(self):
        eeg, marker = recording()
        with self.assertRaises(ValueError):          # not a whole number of epochs
            cut_samples(eeg[:-10], marker[:-10], 1, CONFIG)
        without_imagery = marker.copy()
        without_imagery[4000 + 1] = 0
        with self.assertRaises(ValueError):
            cut_samples(eeg, without_imagery, 1, CONFIG)
        without_start = marker.copy()
        without_start[4000] = 0
        with self.assertRaises(ValueError):
            cut_samples(eeg, without_start, 1, CONFIG)
        with self.assertRaises(ValueError):          # 300 Hz does not divide 500 Hz
            cut_samples(eeg, marker, 1, replace(CONFIG, sample_rate=300))
        with fake_recordings() as reader:            # no trial with room for both samples
            reader.side_effect = lambda path: recording(onsets=(1830, 1900))
            with self.assertRaises(ValueError):
                read_samples("folder", 1, CONFIG)

    def test_alignment_gives_identity_mean_covariance_without_labels(self):
        aligned = align_subject(self.samples, CONFIG)
        windows = [analysed_signal(s, CONFIG) for s in aligned]
        np.testing.assert_allclose(np.mean([w.T @ w / len(w) for w in windows], axis=0), np.eye(29), atol=1e-8)
        self.assertEqual([s.label for s in aligned], [s.label for s in self.samples])
        relabelled = [replace(s, label=1 - s.label) for s in self.samples]
        for a, b in zip(aligned, align_subject(relabelled, CONFIG)):
            np.testing.assert_array_equal(a.signal, b.signal)

    def test_alignment_of_a_rank_deficient_recording(self):
        reduced = [replace(s, signal=s.signal - s.signal.mean(axis=1, keepdims=True)) for s in self.samples]
        windows = [analysed_signal(s, CONFIG) for s in align_subject(reduced, CONFIG)]
        values = np.linalg.eigvalsh(np.mean([w.T @ w / len(w) for w in windows], axis=0))
        np.testing.assert_allclose(values, [0.0] + [1.0] * 28, atol=1e-6)     # identity on the 28 dimensions left

    def test_graph_contents(self):
        with fake_recordings():
            graphs = build_subject_graphs("folder", 1, CONFIG)
        self.assertEqual([int(g.y) for g in graphs], [1, 0] * 4)
        graph = graphs[0]
        self.assertEqual((tuple(graph.x.shape), tuple(graph.edge_index.shape), tuple(graph.edge_attr.shape),
                          tuple(graph.signal.shape), tuple(graph.roi.shape)),
                         ((29, 6), (2, 812), (812, 2), (29, 475), (1, 4)))
        self.assertEqual((graph.subject_id, graph.trial_id, graph.sample_id), (1, 0, 0))
        self.assertFalse((graph.edge_index[0] == graph.edge_index[1]).any())
        self.assertTrue((graph.x > 0).all())
        dense = torch.zeros(29, 29, 2)
        dense[graph.edge_index[0], graph.edge_index[1]] = graph.edge_attr
        torch.testing.assert_close(dense, dense.transpose(0, 1))            # both features are symmetric
        self.assertTrue(((dense[..., 0] >= 0) & (dense[..., 0] <= 1)).all())  # phase-locking value
        signal = graph.signal.double()
        torch.testing.assert_close(dense[..., 1].double().fill_diagonal_(0),
                                   (signal @ signal.T / 475).fill_diagonal_(0), atol=1e-5, rtol=1e-4)
        # A graph depends on the subject's other samples only through the alignment.
        with fake_recordings():
            samples = read_samples("folder", 1, CONFIG)
        alone = build_graph(align_subject(samples, CONFIG)[0], CONFIG)
        torch.testing.assert_close(alone.x, graph.x)

    def test_motor_area_features(self):
        features = roi_features(self.samples, CONFIG)
        self.assertEqual(features.shape, (8, 4))
        np.testing.assert_allclose(features.mean(axis=0), 0, atol=1e-9)        # standardised within the subject
        np.testing.assert_allclose(features.std(axis=0), 1, atol=1e-9)
        left = [EEG_CHANNELS.index(name) for name in ("FC3", "C3", "CP3")]
        louder = self.samples[0].signal.copy()
        louder[:, left] *= 2
        changed = roi_features([replace(self.samples[0], signal=louder)] + self.samples[1:], CONFIG)
        self.assertGreater(changed[0, 0], features[0, 0])                       # left power
        self.assertGreater(changed[0, 3], features[0, 3])                       # left minus right

    @unittest.skipUnless(HAVE_REAL_DATA, "Liu 2024 recordings are not on this machine")
    def test_real_recordings(self):
        graphs = build_subject_graphs(REAL_DATA, 1)                              # the default dataset
        self.assertEqual((len(graphs), sum(int(g.y) for g in graphs)), (80, 40))
        covariance = np.mean([(g.signal.double() @ g.signal.double().T / 475).numpy() for g in graphs], axis=0)
        np.testing.assert_allclose(covariance, np.eye(29), atol=1e-4)           # alignment
        self.assertEqual(len(read_samples(REAL_DATA, 40, CONFIG)), 64)    # eight late trials are dropped
        # The recordings that are left out as duplicates are copies of the ones that are kept.
        for copy, original in LIU2024_DUPLICATES.items():
            a, _ = read_recording(subject_file(REAL_DATA, copy, CONFIG))
            b, _ = read_recording(subject_file(REAL_DATA, original, CONFIG))
            self.assertGreater(np.corrcoef(a[:, 0], b[:, 0])[0, 1], 0.9999)
        self.assertEqual(len(cut_samples(*read_recording(subject_file(REAL_DATA, 6, CONFIG)), 6, CONFIG)), 2)


if __name__ == "__main__":
    unittest.main()
