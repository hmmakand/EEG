"""Recordings -> trials -> aligned trials -> graphs."""
import tempfile
import unittest

import numpy as np
import torch

from ..data import ALL_SUBJECTS, DataConfig, align_subject, build_graph, build_subject_graphs, read_trials
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
        self.assertEqual((config.trial_samples, config.window_samples), (1000, (500, 1000)))
        self.assertEqual(config.feature_bands, ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)))

    def test_trial_selection_and_order(self):
        # Sessions 1 and 2; left/right only; feet and the cut-off trial are skipped.
        self.assertEqual([(t.session_id, t.trial_id, t.label) for t in self.trials],
                         [(s, i, label) for s in (1, 2) for i, label in ((0, 0), (1, 1), (3, 1), (4, 0))])
        self.assertTrue(all(t.signal.shape == (1000, 22) and t.subject_id == 1 for t in self.trials))
        with self.assertRaises(ValueError):
            read_trials(self.folder.name, 1, DataConfig(subjects=(1,), session_indices=(7,)))
        # Without explicit indices every session that contains trials is read.
        every = read_trials(self.folder.name, 1, DataConfig(subjects=(1,)))
        self.assertEqual(sorted({t.session_id for t in every}), [0, 1, 2])
        self.assertEqual(len(every), 12)

    def test_alignment_gives_identity_mean_covariance_without_labels(self):
        aligned = align_subject(self.trials, CONFIG)
        windows = [analysed_signal(t.signal, CONFIG) for t in aligned]
        self.assertEqual(windows[0].shape, (500, 22))
        np.testing.assert_allclose(np.mean([w.T @ w / len(w) for w in windows], axis=0), np.eye(22), atol=1e-8)
        self.assertEqual([t.label for t in aligned], [t.label for t in self.trials])
        relabelled = [type(t)(t.signal, 1 - t.label, t.subject_id, t.session_id, t.trial_id) for t in self.trials]
        for a, b in zip(aligned, align_subject(relabelled, CONFIG)):
            np.testing.assert_array_equal(a.signal, b.signal)

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
