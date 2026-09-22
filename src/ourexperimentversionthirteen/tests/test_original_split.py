"""Original EEG integrity, split reproducibility, and saved-partition reuse."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixture_data import write_subject
from dataset.data_io import load_mat_file, read_sessions
from datasetsynthetic.config import SplitConfig, PARTITIONS
from datasetsynthetic.original import (partition_counts, split_indices,
    create_original_split, load_partition, verify_saved)


class OriginalSplitTests(unittest.TestCase):
    def test_rounding_and_disjoint_class_balanced_partitions(self):
        self.assertEqual(partition_counts(60), [42, 9, 9])
        self.assertEqual(partition_counts(72), [50, 11, 11])
        with self.assertRaises(ValueError):
            partition_counts(2)
        labels = np.array([0] * 60 + [1] * 60)
        config = SplitConfig()
        parts = split_indices(labels, 1, config)
        self.assertEqual(parts, split_indices(labels, 1, config))
        self.assertNotEqual(parts, split_indices(labels, 1, replace(config, seed=43)))
        self.assertEqual(sorted(i for indices in parts.values() for i in indices), list(range(120)))
        for name, expected in zip(PARTITIONS, (42, 9, 9)):
            self.assertEqual(np.bincount(labels[parts[name]]).tolist(), [expected, expected])
            self.assertFalse(np.all(np.diff(labels[parts[name]]) >= 0))
        odd = split_indices([0] * 61 + [1] * 60, 1, config)
        self.assertEqual(sum(map(len, odd.values())), 121)

    def test_original_samples_identity_reuse_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_subject(root)
            output = root / 'original'
            config = SplitConfig(subjects=(1,))
            metadata = create_original_split(root, output, config)
            self.assertEqual(metadata['counts']['1']['training'], {'0': 4, '1': 4})
            sessions = list(read_sessions(load_mat_file(root / 'A01T.mat')))
            seen = set()
            for partition in PARTITIONS:
                signals, labels, rows = load_partition(output, 1, partition)
                self.assertEqual(signals.shape[1:], (1000, 22))
                for signal, label, row in zip(signals, labels, rows):
                    session_id, trial_id = int(row['session_id']), int(row['original_trial_index'])
                    identity = (session_id, trial_id)
                    self.assertNotIn(identity, seen)
                    seen.add(identity)
                    start = int(row['start_sample'])
                    np.testing.assert_array_equal(signal, sessions[session_id].signal[start:start + 1000, :22])
                    self.assertEqual(label, int(row['original_class_label']) - 1)
            self.assertEqual(len(seen), 12)
            mtimes = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
            create_original_split(root, output, config)
            self.assertEqual(mtimes, {p.name: p.stat().st_mtime_ns for p in output.iterdir()})
            with self.assertRaisesRegex(ValueError, 'different config'):
                create_original_split(root, output, replace(config, seed=43))
            with (root / 'A01T.mat').open('ab') as stream:
                stream.write(b'changed source')
            with self.assertRaisesRegex(ValueError, 'source_sha256'):
                create_original_split(root, output, config)
            with (output / 'partitions.csv').open('a') as stream:
                stream.write('tampered\n')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                verify_saved(output)

    def test_incomplete_trial_recorded_without_losing_odd_total(self):
        with tempfile.TemporaryDirectory() as directory:
            from scipy.io import savemat
            root = Path(directory)
            write_subject(root)
            data = load_mat_file(root / 'A01T.mat')
            data[0, 3][0, 0][1][1, 0] = 2000
            savemat(root / 'A01T.mat', {'data': data})
            metadata = create_original_split(root, root / 'original', SplitConfig(subjects=(1,)))
            self.assertEqual(metadata['exclusions'], [dict(subject_id=1, session_id=3,
                original_trial_index=1, original_class_label=2, reason='incomplete_trial')])
            with np.load(root / 'original/subject_01.npz') as saved:
                self.assertEqual(len(saved['signals']), 11)


if __name__ == '__main__':
    unittest.main()
