"""Small CPU integration tests; full dimensions are checked by the CUDA smoke run."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from eegdiffuser.config import Config
from eegdiffuser.data import (selected_signals, training_rows, read_rows, write_rows,
                             subject_loaders, validate_internal_partitions)
from eegdiffuser.generate import generate_subject
from eegdiffuser.runtime import sha256
from eegdiffuser.train import run_training


def fixture(folder: Path):
    folder.mkdir()
    rng = np.random.default_rng(12)
    signals = rng.normal(size=(16, 10, 2)) * 10
    signals[12:] = np.nan  # Held-out original signals must never reach the model.
    labels = np.tile([0, 1], 8)
    np.savez_compressed(folder / 'subject_01.npz', signals=signals, labels=labels)
    rows = []
    for index, label in enumerate(labels):
        partition = 'training' if index < 12 else ('validation' if index < 14 else 'testing')
        rows.append(dict(source_file='A01T.mat', subject_id='1', session_id='3',
                         original_trial_index=str(index), class_label=str(label), array_index=str(index),
                         partition=partition, partition_index=str(index), split_seed='42'))
    write_rows(folder / 'partitions.csv', rows)
    metadata = {'artifact_sha256': {p.name: sha256(p) for p in folder.iterdir()}}
    (folder / 'metadata.json').write_text(json.dumps(metadata))
    return signals


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original = self.root / 'original'
        self.signals = fixture(self.original)
        self.config = replace(Config(), subjects=(1,), epochs=2, checkpoint_interval=1,
                              validation_per_class=1, time_points=10, in_channels=2,
                              embed_dim=16, depth=1, num_heads=2, device='cpu',
                              diffusion_steps=100, samples_per_class=1)

    def tearDown(self):
        self.temporary.cleanup()

    def test_resume_matches_uninterrupted_and_generation_reuses_best(self):
        full, resumed = self.root / 'full', self.root / 'resumed'
        run_training(self.original, full, self.config)
        run_training(self.original, resumed, self.config, stop_after_epoch=1)
        # Simulate interruption after an unsaved epoch replaced best.pt.
        best_path = resumed / 'checkpoints/subject_01/best.pt'
        best_path.write_bytes(b'incomplete newer checkpoint')
        run_training(self.original, resumed, self.config, resume=True)
        def checkpoint(run):
            return torch.load(run / 'checkpoints/subject_01/latest.pt', weights_only=False)
        a, b = checkpoint(full), checkpoint(resumed)
        self.assertEqual(a['epoch'], 2)
        for name in a['model']:
            torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
            torch.testing.assert_close(a['ema'][name], b['ema'][name], rtol=0, atol=0)
        self.assertEqual([r['validation_loss'] for r in a['history']], [r['validation_loss'] for r in b['history']])
        best_epoch = min(a['history'], key=lambda row: row['validation_loss'])['epoch']
        self.assertEqual(a['best_epoch'], best_epoch)
        generate_subject(full, 1, self.config)
        with np.load(full / 'subject_01.npz') as generated:
            self.assertEqual(generated['signals'].shape, (2, 10, 2))
            self.assertTrue(np.isfinite(generated['signals']).all())
            np.testing.assert_array_equal(generated['labels'], [0, 1])
        metadata = json.loads((full / 'metadata.json').read_text())
        self.assertEqual(metadata['generation']['1']['checkpoint_epoch'], best_epoch)
        mtimes = {p.name: p.stat().st_mtime_ns for p in full.iterdir() if p.is_file()}
        generate_subject(full, 1, self.config)
        self.assertEqual(mtimes, {p.name: p.stat().st_mtime_ns for p in full.iterdir() if p.is_file()})
        # The same checkpoint and generation seed reproduce identical signals.
        generate_subject(resumed, 1, self.config)
        with np.load(full / 'subject_01.npz') as first, np.load(resumed / 'subject_01.npz') as second:
            np.testing.assert_array_equal(first['signals'], second['signals'])

    def test_split_scaling_provenance_and_training_mode(self):
        run = self.root / 'run'
        from eegdiffuser import train
        seen_modes = []
        real_make_model = train.make_model
        def monitored_model(config):
            model = real_make_model(config)
            model.register_forward_pre_hook(lambda module, args: seen_modes.append(module.training))
            return model
        with patch.object(train, 'make_model', side_effect=monitored_model):
            run_training(self.original, run, self.config)
        # Ten training and two validation batches per epoch, repeated twice.
        self.assertEqual(seen_modes, ([True] * 10 + [False] * 2) * 2)
        rows = read_rows(run / 'internal_partitions.csv')
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row['partition'] == 'training' for row in rows))
        self.assertEqual(len({row['array_index'] for row in rows}), 12)
        for subset in ('training', 'validation'):
            expected = 5 if subset == 'training' else 1
            for label in (0, 1):
                self.assertEqual(sum(r['diffusion_subset'] == subset and int(r['class_label']) == label for r in rows), expected)
        ordered_indices = [11, 0, 7]
        np.testing.assert_array_equal(selected_signals(self.original / 'subject_01.npz', ordered_indices), self.signals[ordered_indices])
        training_loader, validation_loader = subject_loaders(self.original, run, 1, self.config)
        ordered = sorted((r for r in rows if r['diffusion_subset'] == 'validation'), key=lambda r: int(r['diffusion_index']))
        expected = (self.signals[[int(r['array_index']) for r in ordered]] / 100).transpose(0, 2, 1).astype(np.float32)
        np.testing.assert_array_equal(validation_loader.dataset.tensors[0].numpy(), expected)
        self.assertEqual(len(training_loader.dataset), 10)
        with self.assertRaisesRegex(ValueError, 'settings'):
            run_training(self.original, run, replace(self.config, seed=99), resume=True)
        rows[0]['array_index'] = '15'
        write_rows(run / 'internal_partitions.csv', rows)
        with self.assertRaisesRegex(ValueError, 'changed'):
            run_training(self.original, run, self.config, resume=True)
        with self.assertRaisesRegex(ValueError, 'changed original'):
            validate_internal_partitions(self.original, run / 'internal_partitions.csv', self.config)


if __name__ == '__main__':
    unittest.main()
