"""The Jia 2019 reader, and a run on a dataset without channel names."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io as sio

from ..datascripts import DataConfig, build_subject_graphs, dataset_config, jia2019_config, read_samples, subject_file
from ..datascripts.jia2019 import class_file, resample
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
        self.assertEqual((CONFIG.trial_samples, CONFIG.window_samples, CONFIG.cue_seconds), (1708, 1708, 0.0))
        self.assertTrue(CONFIG.average_reference)
        self.assertEqual(dataset_config("jia2019"), CONFIG)
        model = ModelConfig.for_data(CONFIG)
        self.assertEqual((model.num_electrodes, model.signal_samples), (63, 1708))
        self.assertEqual(subject_file("folder", 7, CONFIG), Path("folder") / "exp1-S7-left.mat")

    def test_resampling_keeps_the_signal_and_gives_volts(self):
        time = np.arange(3500) / 512
        epoch = np.stack([np.sin(2 * np.pi * 10 * time), np.cos(2 * np.pi * 20 * time)] * 2)   # microvolts
        trial = resample(epoch, DataConfig(dataset="jia2019", num_channels=4, trial_seconds=6.832))
        self.assertEqual(trial.shape, (1708, 4))
        new_time = np.arange(1708) / 250
        inner = slice(50, -50)                                # the ends are affected by the resampling filter
        np.testing.assert_allclose(trial[inner, 0], 1e-6 * np.sin(2 * np.pi * 10 * new_time)[inner], atol=2e-8)
        np.testing.assert_allclose(trial[inner, 1], 1e-6 * np.cos(2 * np.pi * 20 * new_time)[inner], atol=2e-8)
        with self.assertRaises(ValueError):
            resample(epoch[:, :3000], CONFIG)

    def test_samples_labels_and_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            write_subject(folder, 2, samples=3500)
            write_subject(folder, 9, samples=4000)             # the longer epochs are cut to the same length
            for subject in (2, 9):
                samples = read_samples(folder, subject, CONFIG)
                self.assertEqual([s.label for s in samples], [0, 0, 0, 1, 1, 1] * 2)   # block 1, then block 2
                self.assertEqual([s.trial_id for s in samples], list(range(12)))
                self.assertTrue(all(s.signal.shape == (1708, 63) and s.start == 0 and s.subject_id == subject
                                    for s in samples))
                # Common average reference: the mean over the electrodes is zero at every time point.
                np.testing.assert_allclose(samples[0].signal.mean(axis=1), 0, atol=1e-12)
                left = np.mean([s.signal.std() for s in samples if s.label == 0])
                right = np.mean([s.signal.std() for s in samples if s.label == 1])
                self.assertAlmostEqual(left / right, 10.0, delta=0.5)      # the label follows the file
            graph = build_subject_graphs(folder, 2, CONFIG)[0]
            self.assertFalse(hasattr(graph, "roi"))                         # no channel names, no motor areas
            os.remove(class_file(folder, 2, "right"))
            with self.assertRaises(ValueError):
                read_samples(folder, 2, CONFIG)
            write_subject(folder, 3, channels=62)
            with self.assertRaises(ValueError):
                read_samples(folder, 3, CONFIG)

    def test_leave_one_subject_out_runs_without_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            for subject in (1, 2, 3):
                write_subject(folder, subject, trials=2, channels=5)
            config = DataConfig(dataset="jia2019", class_names=("left hand", "right hand"), subjects=(1, 2, 3),
                                num_channels=5, trial_seconds=6.832, sample_seconds=3.5, cue_seconds=3.0)
            result = run_loso(folder, config, TrainConfig(epochs=2, batch_size=4, ensemble=1, device="cpu",
                                                          show_progress=False), held_out_subjects=(1,))
            self.assertFalse(result["train_config"]["target_calibration"])
            self.assertNotIn("calibration", result["subjects"]["1"])

    @unittest.skipUnless(HAVE_REAL_DATA, "Jia 2019 recordings are not on this machine")
    def test_real_recording(self):
        samples = read_samples(REAL_DATA, 1, CONFIG)
        self.assertEqual((len(samples), sum(s.label for s in samples)), (80, 40))
        self.assertTrue(1e-6 < np.mean([s.signal.std() for s in samples]) < 1e-4)     # volts
        graph = build_subject_graphs(REAL_DATA, 1, CONFIG)[0]
        self.assertEqual((tuple(graph.x.shape), tuple(graph.edge_index.shape), tuple(graph.edge_attr.shape),
                          tuple(graph.signal.shape)), ((63, 6), (2, 3906), (3906, 2), (63, 1708)))


if __name__ == "__main__":
    unittest.main()
