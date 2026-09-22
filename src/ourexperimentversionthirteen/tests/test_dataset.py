"""Run with python -m unittest discover -s tests from the experiment folder."""
from dataclasses import replace
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import build_subject_dataset, default_dataset_config
from dataset.config import GraphConfig
from dataset.graphs import connectivity_to_edges
from dataset.trials import Trial, order_trials, extract_trials
from dataset.data_io import Session
from dataset.validation import validate_config
from model.gat import GAT
from fixture_data import write_subject


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        write_subject(cls.temp.name)
        cls.config = default_dataset_config()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_original_pipeline_equivalence(self):
        with np.load(Path(__file__).parent / 'fixtures/legacy_expected.npz') as expected:
            for name, threshold in [('zero', 0), ('positive', 0.1)]:
                config = replace(self.config, graphs=replace(self.config.graphs, threshold=threshold),
                                 connectivity=replace(self.config.connectivity, method="plv"))
                graphs = build_subject_dataset(self.temp.name, 1, config)
                np.testing.assert_allclose(np.stack([g.x.numpy() for g in graphs]), expected[name+'_x'], rtol=1e-6, atol=1e-9)
                np.testing.assert_array_equal(np.stack([g.edge_index.numpy() for g in graphs]), expected[name+'_edges'])
                np.testing.assert_array_equal([g.y.item() for g in graphs], expected[name+'_y'])
                self.assertEqual([g.trial_id for g in graphs], [0, 1] * 6)

    def test_configurable_features_and_training(self):
        config = replace(self.config, features=replace(self.config.features, band_edges=(8, 12, 16)),
                         trials=replace(self.config.trials, channel_indices=(0, 2, 4)))
        graphs = build_subject_dataset(self.temp.name, 1, config)
        batch = Batch.from_data_list(graphs[:2])
        model = GAT(hidden_channels=4, heads=2, in_channels=graphs[0].num_node_features)
        output = model(batch.x, batch.edge_index, batch.batch)
        self.assertEqual(tuple(output.shape), (2, 2))
        torch.nn.functional.cross_entropy(output, batch.y).backward()
        self.assertEqual(tuple(graphs[0].x.shape), (3, 2))

    def test_trainer_integration(self):
        from train.config import default_training_config
        from train.experiment import run_subject
        training_config = replace(default_training_config(), n_folds=2, num_epochs=1, show_progress=False)
        results = run_subject(self.temp.name, 1, torch.device('cpu'), self.config, training_config)
        self.assertTrue({'mean', 'max', 'min', 'balanced_accuracy', 'macro_f1', 'confusion_matrix', 'folds'} <= set(results))
        self.assertTrue(all(0 <= results[key] <= 1 for key in ('mean', 'max', 'min')))

    def test_threshold_edges_and_legacy_placeholders(self):
        matrix = np.array([[0, .5, .9], [.5, 0, .2], [.9, .2, 0]])
        edges, weights = connectivity_to_edges(matrix, GraphConfig(mode='threshold', threshold=.5))
        np.testing.assert_array_equal(edges.numpy(), [[0, 2], [2, 0]])
        np.testing.assert_allclose(weights.numpy(), [.9, .9])
        edges, _ = connectivity_to_edges(matrix, GraphConfig(threshold=.5))
        self.assertEqual(edges.shape[1], 9)
        self.assertEqual(int((edges == 0).all(dim=0).sum()), 7)
        edges, _ = connectivity_to_edges(matrix, GraphConfig(mode='threshold', threshold=1))
        self.assertEqual(tuple(edges.shape), (2, 0))

    def test_unequal_class_ordering(self):
        items = [Trial(np.ones((10, 2)), label, 1, 0, i) for i, label in enumerate([0, 0, 0, 1, 1])]
        self.assertEqual([t.trial_id for t in order_trials(items, 'legacy_interleave')], [0, 2, 1, 3])
        self.assertEqual([t.trial_id for t in order_trials(items, 'interleave')], [0, 3, 1, 4, 2])

    def test_trial_selection_and_index_offset(self):
        signal = np.arange(3000).reshape(1500, 2)
        session = Session(signal, np.array([1, 1400]), np.array([2, 1]), 0)
        config = replace(self.config.trials, session_indices=(0,), channel_indices=(1,), event_index_offset=-1)
        trials = extract_trials([session], 7, 250, config)
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0].label, 1)
        np.testing.assert_array_equal(trials[0].signal[:, 0], signal[:1000, 1])
        with self.assertRaisesRegex(ValueError, 'Incomplete trial'):
            extract_trials([session], 7, 250, replace(config, incomplete_policy='error'))

    def test_invalid_config(self):
        for config in [replace(self.config, sample_rate=0),
                       replace(self.config, features=replace(self.config.features, band_edges=(12, 8))),
                       replace(self.config, graphs=replace(self.config.graphs, self_loops=True))]:
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_standalone_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'copied_experiment'
            shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns('__pycache__', 'output', 'tests'))
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            for command in ([sys.executable, str(target / 'train/train.py'), '--help'],
                            [sys.executable, '-m', 'train.train', '--help']):
                result = subprocess.run(command, cwd=target, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--data-dir', result.stdout)
            output = target / 'output' / 'smoke.json'
            env['OMP_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            result = subprocess.run(
                [sys.executable, str(target / 'train/train.py'),
                 '--data-dir', self.temp.name, '--subjects', '1', '--folds', '2',
                 '--epochs', '1', '--device', 'cpu', '--output', str(output)],
                cwd=directory, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            results = json.loads(output.read_text())
            self.assertEqual(set(results), {'1'})
            self.assertEqual(len(results['1']['folds']), 2)
            self.assertTrue(all(fold['selected_epoch'] == 1 for fold in results['1']['folds']))
            self.assertEqual(sum(map(sum, results['1']['confusion_matrix'])), 12)


if __name__ == '__main__':
    unittest.main()
