"""The Jia 2019 reader: files, blocks, labels and resampling."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io as sio

from ..data import DataConfig, build_subject_graphs, dataset_config, jia2019_config, read_trials, subject_file
from ..data.jia2019 import class_file, resample
from ..model import ModelConfig
from ..train import TrainConfig, run_loso

CONFIG = jia2019_config()
# Set JIA2019_DATA to another location to run the checks on the real recordings there.
REAL_DATA = Path(os.environ.get("JIA2019_DATA", "/workspaces/datadisk/databci/eegdata/jia2019"))
HAVE_REAL_DATA = all(class_file(REAL_DATA, 1, side).is_file() for side in ("left", "right"))


def write_subject(folder, subject, samples=3500, trials=3, channels=63):
    """Both class files of one subject; the left file is ten times larger in amplitude."""
    rng = np.random.default_rng(subject)
    for side, scale in (("left", 10.0), ("right", 1.0)):
        blocks = {}
        for key in ("DATA1", "DATA2"):
            cell = np.empty((1, trials), dtype=object)
            for trial in range(trials):
                cell[0, trial] = rng.normal(size=(channels, samples)) * scale
            blocks[key] = cell
        sio.savemat(class_file(folder, subject, side), blocks)


class Jia2019Tests(unittest.TestCase):
    def test_rules(self):
        self.assertEqual((CONFIG.dataset, CONFIG.subjects, CONFIG.num_channels), ("jia2019", tuple(range(1, 16)), 63))
        self.assertEqual((CONFIG.trial_samples, CONFIG.window_offsets, CONFIG.window_samples), (1708, (0, 875), 875))
        self.assertEqual(dataset_config("jia2019"), CONFIG)
        model = ModelConfig.for_data(CONFIG)
        self.assertEqual((model.num_electrodes, model.signal_samples), (63, 875))
        self.assertEqual(subject_file("folder", 7, CONFIG), Path("folder") / "exp1-S7-left.mat")

    def test_resampling_keeps_the_signal_and_gives_volts(self):
        time = np.arange(3500) / 512
        epoch = np.stack([np.sin(2 * np.pi * 10 * time), np.cos(2 * np.pi * 20 * time)] * 2)   # microvolts
        trial = resample(epoch, DataConfig(dataset="jia2019", subjects=(1,), num_channels=4, trial_seconds=6.832))
        self.assertEqual(trial.shape, (1708, 4))
        new_time = np.arange(1708) / 250
        inner = slice(50, -50)                                # the ends are affected by the resampling filter
        np.testing.assert_allclose(trial[inner, 0], 1e-6 * np.sin(2 * np.pi * 10 * new_time)[inner], atol=2e-8)
        np.testing.assert_allclose(trial[inner, 1], 1e-6 * np.cos(2 * np.pi * 20 * new_time)[inner], atol=2e-8)
        with self.assertRaises(ValueError):
            resample(epoch[:, :3000], CONFIG)

    def test_trials_labels_and_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            write_subject(folder, 2, samples=3500)
            write_subject(folder, 9, samples=4000)             # the longer epochs are cut to the same length
            for subject in (2, 9):
                trials = read_trials(folder, subject, CONFIG)
                self.assertEqual([(t.session_id, t.label, t.trial_id) for t in trials],
                                 [(block, label, i) for block in (0, 1) for label in (0, 1) for i in range(3)])
                self.assertTrue(all(t.signal.shape == (1708, 63) and t.onset == 750 and t.subject_id == subject
                                    for t in trials))
                left = np.mean([t.signal.std() for t in trials if t.label == 0])
                right = np.mean([t.signal.std() for t in trials if t.label == 1])
                self.assertAlmostEqual(left / right, 10.0, delta=0.5)      # the label follows the file
            os.remove(class_file(folder, 2, "right"))
            with self.assertRaises(ValueError):
                read_trials(folder, 2, CONFIG)
            write_subject(folder, 3, channels=62)
            with self.assertRaises(ValueError):
                read_trials(folder, 3, CONFIG)

    def test_leave_one_subject_out_runs_without_calibration(self):
        # jia2019 has no channel names, so the default target calibration is switched off for it.
        with tempfile.TemporaryDirectory() as folder:
            for subject in (1, 2, 3):
                write_subject(folder, subject, trials=2, channels=5)
            config = DataConfig(dataset="jia2019", subjects=(1, 2, 3), num_channels=5, trial_seconds=6.832,
                                window_seconds=(0.0, 3.5))
            result = run_loso(folder, config, config=TrainConfig(epochs=2, batch_size=4, ensemble=1, device="cpu",
                                                                  show_progress=False), held_out_subjects=(1,))
            self.assertFalse(result["train_config"]["target_calibration"])
            self.assertNotIn("calibration", result["subjects"]["1"])

    @unittest.skipUnless(HAVE_REAL_DATA, "Jia 2019 recordings are not on this machine")
    def test_real_recording(self):
        trials = read_trials(REAL_DATA, 1, CONFIG)
        self.assertEqual((len(trials), sum(t.label for t in trials)), (80, 40))
        self.assertTrue(1e-6 < np.mean([t.signal.std() for t in trials]) < 1e-4)     # volts
        graph = build_subject_graphs(REAL_DATA, 1, CONFIG)[0]
        self.assertEqual((tuple(graph.x.shape), tuple(graph.edge_index.shape), tuple(graph.edge_attr.shape),
                          tuple(graph.signal.shape)), ((63, 6), (2, 3906), (3906, 2), (63, 875)))


if __name__ == "__main__":
    unittest.main()
