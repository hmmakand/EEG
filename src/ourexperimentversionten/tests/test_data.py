"""Recordings -> trials -> aligned trials -> graphs."""
import tempfile
import unittest

import numpy as np
import torch

from ..data import ALL_SUBJECTS, DataConfig, align_subject, build_graph, build_subject_graphs, read_trials
from ..data.graph import phase_locking, plv_edges
from ..data.signal import analysed_signal
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
        config = DataConfig()
        self.assertEqual(config.subjects, (1, 2, 3, 5, 6, 7, 8, 9))
        self.assertEqual(ALL_SUBJECTS, (1, 2, 3, 4, 5, 6, 7, 8, 9))
        self.assertIsNone(config.session_indices)
        self.assertEqual((config.trial_samples, config.window_samples, config.plv_threshold), (1000, (500, 1000), 0.35))
        self.assertEqual(config.feature_bands, ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)))

    def test_trial_selection_and_order(self):
        # Sessions 1 and 2; left/right only; feet and the cut-off trial are skipped.
        self.assertEqual([(t.session_id, t.trial_id, t.label) for t in self.trials],
                         [(s, i, label) for s in (1, 2) for i, label in ((0, 0), (1, 1), (3, 1), (4, 0))])
        self.assertTrue(all(t.signal.shape == (1000, 22) and t.subject_id == 1 for t in self.trials))
        every = read_trials(self.folder.name, 1, DataConfig(subjects=(1,)))
        self.assertEqual(sorted({t.session_id for t in every}), [0, 1, 2])
        with self.assertRaises(ValueError):
            read_trials(self.folder.name, 1, DataConfig(subjects=(1,), session_indices=(7,)))

    def test_alignment_gives_identity_mean_covariance_without_labels(self):
        aligned = align_subject(self.trials, CONFIG)
        windows = [analysed_signal(t.signal, CONFIG) for t in aligned]
        self.assertEqual(windows[0].shape, (500, 22))
        np.testing.assert_allclose(np.mean([w.T @ w / len(w) for w in windows], axis=0), np.eye(22), atol=1e-8)
        relabelled = [type(t)(t.signal, 1 - t.label, t.subject_id, t.session_id, t.trial_id) for t in self.trials]
        for a, b in zip(aligned, align_subject(relabelled, CONFIG)):
            np.testing.assert_array_equal(a.signal, b.signal)

    def test_plv_edges(self):
        time = np.arange(500) / 250
        rng = np.random.default_rng(0)
        window = rng.normal(size=(500, 4))
        window[:, 1] = window[:, 0] = np.sin(2 * np.pi * 10 * time)   # channels 0 and 1 are phase-locked
        matrix = phase_locking(window)
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), np.zeros(4))
        self.assertAlmostEqual(matrix[0, 1], 1.0, places=6)
        self.assertEqual(plv_edges(window, 0.9).T.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(plv_edges(window, 1.5).shape, (2, 0))

    def test_graph_contents(self):
        graphs = build_subject_graphs(self.folder.name, 1, CONFIG)
        self.assertEqual(len(graphs), 8)
        graph = graphs[0]
        self.assertEqual(tuple(graph.x.shape), (22, 6))
        self.assertTrue(torch.isfinite(graph.x).all())
        self.assertEqual((int(graph.y), graph.subject_id, graph.session_id, graph.trial_id), (0, 1, 1, 0))
        self.assertFalse((graph.edge_index[0] == graph.edge_index[1]).any())           # no self-loops
        pairs = set(map(tuple, graph.edge_index.T.tolist()))
        self.assertTrue(all((j, i) in pairs for i, j in pairs))                        # both directions
        # Node features come from the aligned signal, edges from the unaligned one.
        aligned = align_subject(self.trials, CONFIG)[0]
        unaligned_features = build_graph(self.trials[0], self.trials[0], CONFIG).x
        self.assertFalse(torch.allclose(unaligned_features, graph.x))
        expected = plv_edges(analysed_signal(self.trials[0].signal, CONFIG), CONFIG.plv_threshold)
        np.testing.assert_array_equal(build_graph(self.trials[0], aligned, CONFIG).edge_index.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
