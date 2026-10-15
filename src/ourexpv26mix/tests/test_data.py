"""Recordings -> trials -> aligned trials -> graphs, on recordings in the BCI IV 2a layout."""
from dataclasses import replace
import tempfile
import unittest

import numpy as np
import torch

from ..data import (BCI2A_ALL_SUBJECTS, DataConfig, align_subject, bci2a_config, build_graph, build_subject_graphs,
                    dataset_config, liu2024_config, read_trials)
from ..data import jia2019_config
from ..data.roi import BCI2A_CHANNELS, roi_features
from ..data.graph import covariance_logarithm
from ..data.signal import analysed_signal
from ..model import ModelConfig
from .fixture import CONFIG, write_recordings


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        write_recordings(cls.folder.name)
        cls.trials = read_trials(cls.folder.name, 1, CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_default_rules(self):
        config = bci2a_config()
        self.assertEqual(config.subjects, (1, 2, 3, 5, 6, 7, 8, 9))
        self.assertEqual(bci2a_config(all_subjects=True).subjects, BCI2A_ALL_SUBJECTS)
        self.assertIsNone(config.session_indices)
        self.assertEqual((config.num_channels, config.trial_samples, config.window_offsets, config.window_samples),
                         (22, 1000, (0, 500), 500))
        default = liu2024_config()                                   # imagery against rest
        self.assertEqual((default.task, default.window_samples, len(default.subjects)), ("imagery_rest", 475, 46))
        self.assertEqual(set(range(1, 51)) - set(default.subjects), {5, 6, 15, 49})
        self.assertEqual(dataset_config("liu2024"), default)
        liu = liu2024_config(task="left_right")
        self.assertEqual(liu, DataConfig())
        self.assertEqual((liu.num_channels, liu.trial_samples, liu.window_offsets, liu.window_samples),
                         (29, 2000, (0, 1000), 1000))
        self.assertEqual((len(liu.subjects), set(range(1, 51)) - set(liu.subjects)), (47, {5, 15, 49}))
        self.assertEqual(liu2024_config(all_subjects=True).subjects, tuple(range(1, 51)))
        self.assertEqual(liu2024_config(task="left_right", all_subjects=True).subjects, tuple(range(1, 51)))
        self.assertEqual(dataset_config("bci2a", window_seconds=(0.5, 2.0)).window_offsets, (125, 500))
        for bad in (dict(dataset="other"), dict(subjects=()), dict(subjects=(1, 1)), dict(window_seconds=(2.0, 2.0))):
            with self.assertRaises(ValueError):
                DataConfig(**bad)
        self.assertEqual(config.feature_bands, ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)))

    def test_trial_selection_and_order(self):
        # Sessions 1 and 2; left/right only; feet and the cut-off trial are skipped.
        self.assertEqual([(t.session_id, t.trial_id, t.label) for t in self.trials],
                         [(s, i, label) for s in (1, 2) for i, label in ((0, 0), (1, 1), (3, 1), (4, 0))])
        self.assertTrue(all(t.signal.shape == (1000, 22) and t.subject_id == 1 and t.onset == 500 for t in self.trials))
        with self.assertRaises(ValueError):
            read_trials(self.folder.name, 1, bci2a_config(subjects=(1,), session_indices=(7,)))
        # Without explicit indices every session that contains trials is read.
        every = read_trials(self.folder.name, 1, bci2a_config(subjects=(1,)))
        self.assertEqual(sorted({t.session_id for t in every}), [0, 1, 2])
        self.assertEqual(len(every), 12)

    def test_alignment_gives_identity_mean_covariance_without_labels(self):
        aligned = align_subject(self.trials, CONFIG)
        windows = [analysed_signal(t, CONFIG) for t in aligned]
        self.assertEqual(windows[0].shape, (500, 22))
        np.testing.assert_allclose(np.mean([w.T @ w / len(w) for w in windows], axis=0), np.eye(22), atol=1e-8)
        self.assertEqual([t.label for t in aligned], [t.label for t in self.trials])
        relabelled = [replace(t, label=1 - t.label) for t in self.trials]
        for a, b in zip(aligned, align_subject(relabelled, CONFIG)):
            np.testing.assert_array_equal(a.signal, b.signal)

    def test_alignment_of_a_rank_deficient_recording(self):
        # Remove the mean over the electrodes, as an average reference does: one dimension is lost.
        reduced = [replace(t, signal=t.signal - t.signal.mean(axis=1, keepdims=True)) for t in self.trials]
        aligned = align_subject(reduced, CONFIG)
        windows = [analysed_signal(t, CONFIG) for t in aligned]
        mean_covariance = np.mean([w.T @ w / len(w) for w in windows], axis=0)
        self.assertTrue(np.isfinite(mean_covariance).all())
        values = np.linalg.eigvalsh(mean_covariance)
        np.testing.assert_allclose(values, [0.0] + [1.0] * 21, atol=1e-6)     # identity on the 21 dimensions left
        self.assertLess(np.abs(np.stack(windows)).max(), 100)                 # nothing was blown up

    def test_motor_area_features(self):
        features = roi_features(self.trials, CONFIG)
        self.assertEqual(features.shape, (8, 4))
        np.testing.assert_allclose(features.mean(axis=0), 0, atol=1e-9)        # standardised within the subject
        np.testing.assert_allclose(features.std(axis=0), 1, atol=1e-9)
        # Doubling the left motor electrodes of one trial raises its left power and its left-right difference.
        left = [BCI2A_CHANNELS.index(name) for name in ("FC3", "C3", "CP3")]
        louder = self.trials[0].signal.copy()
        louder[:, left] *= 2
        changed = roi_features([replace(self.trials[0], signal=louder)] + self.trials[1:], CONFIG)
        self.assertGreater(changed[0, 0], features[0, 0])
        self.assertGreater(changed[0, 3], features[0, 3])
        graph = build_subject_graphs(self.folder.name, 1, CONFIG)[0]
        torch.testing.assert_close(graph.roi, torch.tensor(features[:1], dtype=torch.float32))
        with self.assertRaises(ValueError):                                     # jia2019 has no channel names
            roi_features(self.trials, jia2019_config())

    def test_tangent_features(self):
        config = replace(CONFIG, tangent_features=True)
        plain = build_subject_graphs(self.folder.name, 1, CONFIG)[0]
        graph = build_subject_graphs(self.folder.name, 1, config)[0]
        self.assertEqual((tuple(graph.x.shape), tuple(graph.edge_attr.shape)), ((22, 7), (462, 3)))
        torch.testing.assert_close(graph.x[:, :6], plain.x)                      # the other features are untouched
        torch.testing.assert_close(graph.edge_attr[:, :2], plain.edge_attr)
        # The new entries are the logarithm of the covariance of the stored signal.
        signal = graph.signal.double().numpy()
        expected = covariance_logarithm(signal @ signal.T / 500)
        dense = np.zeros((22, 22))
        dense[graph.edge_index[0], graph.edge_index[1]] = graph.edge_attr[:, 2].numpy()
        np.testing.assert_allclose(dense, expected - np.diag(np.diag(expected)), atol=1e-4)
        np.testing.assert_allclose(np.log(graph.x[:, 6].numpy()), np.diag(expected), atol=1e-4)
        # The logarithm of the identity is zero, and exponentiating gives the matrix back.
        np.testing.assert_allclose(covariance_logarithm(np.eye(4)), 0, atol=1e-12)
        matrix = np.cov(np.random.default_rng(0).normal(size=(5, 50)))
        values, vectors = np.linalg.eigh(covariance_logarithm(matrix))
        np.testing.assert_allclose((vectors * np.exp(values)) @ vectors.T, matrix, atol=1e-10)
        model = ModelConfig.for_data(config)
        self.assertEqual((model.node_features, model.edge_features, model.low_samples), (7, 3, 0))

    def test_low_frequency_signal(self):
        config = replace(CONFIG, low_band=(0.5, 4.0))
        self.assertEqual((config.low_samples, CONFIG.low_samples), (50, 0))     # 2 s at 25 Hz
        graphs = build_subject_graphs(self.folder.name, 1, config)
        low = torch.stack([graph.signal_low for graph in graphs])
        self.assertEqual(tuple(low.shape), (8, 22, 50))
        torch.testing.assert_close(low.mean(dim=2), torch.zeros(8, 22), atol=1e-5, rtol=0)   # mean removed per channel
        # The main signal and features are those without the option.
        plain = build_subject_graphs(self.folder.name, 1, CONFIG)
        torch.testing.assert_close(graphs[0].signal, plain[0].signal)
        self.assertFalse(hasattr(plain[0], "signal_low"))
        # The low band is aligned on its own: its mean covariance over the window is the identity.
        aligned = align_subject(self.trials, config, low=True)
        windows = [analysed_signal(t, config, low=True) for t in aligned]
        np.testing.assert_allclose(np.mean([w.T @ w / len(w) for w in windows], axis=0), np.eye(22), atol=1e-8)
        self.assertEqual(ModelConfig.for_data(config).low_samples, 50)
        for bad in ((0.0, 4.0), (4.0, 0.5), (0.5, 13.0)):
            with self.assertRaises(ValueError):
                replace(CONFIG, low_band=bad)

    def test_graph_contents(self):
        graphs = build_subject_graphs(self.folder.name, 1, CONFIG)
        self.assertEqual(len(graphs), 8)
        graph = graphs[0]
        self.assertEqual((tuple(graph.x.shape), tuple(graph.edge_index.shape), tuple(graph.edge_attr.shape),
                          tuple(graph.signal.shape)), ((22, 6), (2, 462), (462, 2), (22, 500)))
        self.assertEqual((int(graph.y), graph.subject_id, graph.session_id, graph.trial_id), (0, 1, 1, 0))
        self.assertFalse((graph.edge_index[0] == graph.edge_index[1]).any())
        self.assertTrue((graph.x > 0).all())
        dense = torch.zeros(22, 22, 2)
        dense[graph.edge_index[0], graph.edge_index[1]] = graph.edge_attr
        torch.testing.assert_close(dense, dense.transpose(0, 1))            # both features are symmetric
        self.assertTrue(((dense[..., 0] >= 0) & (dense[..., 0] <= 1)).all())  # phase-locking value
        # The covariance feature is the covariance of the stored signal.
        signal = graph.signal.double()
        torch.testing.assert_close(dense[..., 1].double().fill_diagonal_(0),
                                   (signal @ signal.T / 500).fill_diagonal_(0), atol=1e-5, rtol=1e-4)
        # A graph depends on the subject's other trials only through the alignment.
        alone = build_graph(align_subject(self.trials, CONFIG)[0], CONFIG)
        torch.testing.assert_close(alone.x, graph.x)


if __name__ == "__main__":
    unittest.main()
