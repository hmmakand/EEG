"""Fixed-partition graph equivalence, isolation, batching, and model selection."""
from contextlib import redirect_stdout
from dataclasses import replace
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixture_data import write_subject
from datasetsynthetic.config import SplitConfig
from datasetsynthetic.original import create_original_split
from dataset import default_dataset_config, build_trial_graph
from dataset.trials import Trial
from datacustom.dataset import build_subject_dataset, sha256
from train.config import default_training_config
from train.config_syn import SynTrainingConfig
from train.splitting_syn import build_loaders, make_augmented_folds
from train import experiment_syn, engine
from train.engine import Evaluation
from train.train_syn import main


def make_synthetic(original, output):
    output.mkdir()
    with np.load(original / 'subject_01.npz') as arrays:
        signals = arrays['signals'][:4].copy() * 0.9
        labels = arrays['labels'][:4].copy()
    identifiers = np.array([f'S01_synthetic_{i:05d}' for i in range(4)])
    np.savez_compressed(output / 'subject_01.npz', signals=signals, labels=labels,
                        subject_ids=np.ones(4, dtype=np.int64), synthetic_trial_ids=identifiers)
    rows = [dict(subject_id='1', synthetic_trial_id=str(identifiers[i]), class_label=str(int(labels[i])),
                 array_index=str(i), output_file='subject_01.npz', source='synthetic',
                 intended_partition='training', checkpoint_sha256='fixture_checkpoint') for i in range(4)]
    with (output / 'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(status='complete', original_partition_used='training', original_validation_and_testing_used=False,
                    original_sha256={name: sha256(original / name) for name in ('metadata.json', 'partitions.csv', 'subject_01.npz')},
                    manifest_sha256=sha256(output / 'manifest.csv'),
                    generation={'1': dict(sha256=sha256(output / 'subject_01.npz'), trial_count=4,
                                          shape=list(signals.shape), checkpoint_sha256='fixture_checkpoint',
                                          counts={str(c): int(sum(labels == c)) for c in (0, 1)})})
    (output / 'metadata.json').write_text(json.dumps(metadata))


class FixedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.original, cls.synthetic = cls.root / 'original', cls.root / 'synthetic'
        write_subject(cls.root)
        with redirect_stdout(io.StringIO()):
            create_original_split(cls.root, cls.original, SplitConfig(subjects=(1,)))
        make_synthetic(cls.original, cls.synthetic)
        cls.baseline = build_subject_dataset(cls.original, subject_number=1)
        cls.augmented = build_subject_dataset(cls.original, cls.synthetic, 1)
        cls.config = SynTrainingConfig(subjects=(1,), num_epochs=1, hidden_channels=3, heads=1,
                                      show_progress=False, device='cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_graphs_match_existing_builder_and_holdout_partitions_match(self):
        self.assertEqual({k: len(v) for k, v in self.baseline.items()}, dict(training=8, validation=2, testing=2))
        self.assertEqual({k: len(v) for k, v in self.augmented.items()}, dict(training=12, validation=2, testing=2))
        with (self.original / 'partitions.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        with np.load(self.original / 'subject_01.npz') as saved:
            for partition, graphs in self.baseline.items():
                selected = sorted((r for r in rows if r['partition'] == partition), key=lambda r: int(r['partition_index']))
                for graph, row in zip(graphs, selected):
                    index = int(row['array_index'])
                    direct = build_trial_graph(Trial(saved['signals'][index], int(saved['labels'][index]),
                        1, int(row['session_id']), int(row['original_trial_index'])))
                    torch.testing.assert_close(graph.x, direct.x, rtol=0, atol=0)
                    torch.testing.assert_close(graph.edge_index, direct.edge_index, rtol=0, atol=0)
                    self.assertEqual(graph.y.item(), direct.y.item())
                if partition != 'training':
                    for a, b in zip(graphs, self.augmented[partition]):
                        self.assertEqual(a.sample_id, b.sample_id)
                        torch.testing.assert_close(a.x, b.x, rtol=0, atol=0)
                        torch.testing.assert_close(a.edge_index, b.edge_index, rtol=0, atol=0)
        synthetic_graphs = [g for g in self.augmented['training'] if g.is_synthetic]
        with np.load(self.synthetic / 'subject_01.npz') as saved:
            for i, graph in enumerate(synthetic_graphs):
                direct = build_trial_graph(Trial(saved['signals'][i], int(saved['labels'][i]), 1, -1, i))
                torch.testing.assert_close(graph.x, direct.x, rtol=0, atol=0)
        batch = Batch.from_data_list(self.augmented['training'])
        self.assertEqual(tuple(batch.x.shape), (12 * 22, 8))
        self.assertEqual(len(batch.sample_id), 12)
        self.assertEqual(int(batch.is_synthetic.sum()), 4)

    def test_loaders_mix_without_resplitting_and_reject_leakage(self):
        before = [g.sample_id for g in self.augmented['training']]
        train, val, test = build_loaders(self.augmented, self.config)
        again, _, _ = build_loaders(self.augmented, self.config)
        baseline, _, _ = build_loaders(self.baseline, self.config)
        self.assertEqual([g.sample_id for g in train.dataset], [g.sample_id for g in again.dataset])
        self.assertEqual([g.sample_id for g in train.dataset if not g.is_synthetic], [g.sample_id for g in baseline.dataset])
        self.assertEqual(before, [g.sample_id for g in self.augmented['training']])
        for loader, name in ((val, 'validation'), (test, 'testing')):
            self.assertEqual([g.sample_id for g in loader.dataset], [g.sample_id for g in self.baseline[name]])
        bad = {k: list(v) for k, v in self.augmented.items()}
        graph = bad['training'].pop().clone()
        graph.partition = 'testing'
        bad['testing'].append(graph)
        with self.assertRaisesRegex(ValueError, 'Synthetic graphs'):
            build_loaders(bad, self.config)
        bad = {k: list(v) for k, v in self.baseline.items()}
        bad['training'].append(bad['training'][0])
        with self.assertRaisesRegex(ValueError, 'identity'):
            build_loaders(bad, self.config)

    def test_validation_selects_first_best_weights_and_test_is_called_once(self):
        class CounterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
        model = CounterModel()
        events = []
        def train(model, *args):
            with torch.no_grad():
                model.weight.add_(1)
            events.append('train')
        def evaluate(model, loader, device, *, collect_predictions=False):
            partition = loader.dataset[0].partition
            events.append(partition)
            if partition == 'testing':
                self.assertEqual(float(model.weight.detach()), 2.0)
                self.assertTrue(collect_predictions)
                return Evaluation(.5, [0, 1], [1, 1])
            if partition == 'validation':
                return [.5, .75, .75][int(model.weight.detach()) - 1]
            return .5
        self.assertIs(experiment_syn.train_epoch, engine.train_epoch)
        self.assertIs(experiment_syn.evaluate, engine.evaluate)
        with patch.object(experiment_syn, 'build_model', return_value=model), \
             patch.object(experiment_syn, 'train_epoch', side_effect=train), \
             patch.object(experiment_syn, 'evaluate', side_effect=evaluate):
            result = experiment_syn.run_subject(self.baseline, torch.device('cpu'), replace(self.config, num_epochs=3))
        self.assertEqual(result['selected_epoch'], 2)
        self.assertEqual(result['best_validation_accuracy'], .75)
        self.assertEqual(result['test']['accuracy'], .5)
        self.assertEqual(events, ['train', 'training', 'validation'] * 3 + ['testing'])

    def test_hash_and_source_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'synthetic'
            shutil.copytree(self.synthetic, target)
            path = target / 'metadata.json'
            metadata = json.loads(path.read_text())
            metadata['original_sha256']['partitions.csv'] = 'different_partitions'
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'different original data'):
                build_subject_dataset(self.original, target, 1)
            metadata['original_sha256']['partitions.csv'] = sha256(self.original / 'partitions.csv')
            path.write_text(json.dumps(metadata))
            with (target / 'manifest.csv').open('a') as stream:
                stream.write('modified')
            with self.assertRaisesRegex(ValueError, 'artifact has changed'):
                build_subject_dataset(self.original, target, 1)

    def test_combined_selection_keeps_earliest_best_metrics_without_extra_test(self):
        for zero_scores in (False, True):
            with self.subTest(zero_scores=zero_scores):
                model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    model.weight.zero_()
                evaluated = []
                def train(model, *args):
                    with torch.no_grad():
                        model.weight.add_(1)
                def evaluate(model, loader, device, *, collect_predictions=False):
                    parts = {g.partition for g in loader.dataset}
                    if parts == {'training'}:
                        evaluated.append('training')
                        return .5
                    self.assertEqual(parts, {'validation', 'testing'})
                    self.assertEqual(len(loader.dataset), 4)
                    self.assertFalse(any(g.is_synthetic for g in loader.dataset))
                    self.assertTrue(collect_predictions)
                    evaluated.append('combined')
                    epoch = int(model.weight.detach().item())
                    predictions = ([1, 0, 1, 0] if zero_scores else
                                   [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1]][epoch - 1])
                    accuracy = 0. if zero_scores else [.5, .75, .75][epoch - 1]
                    return Evaluation(accuracy, [0, 1, 0, 1], predictions)
                with patch.object(experiment_syn, 'build_model', return_value=model), \
                     patch.object(experiment_syn, 'train_epoch', side_effect=train), \
                     patch.object(experiment_syn, 'evaluate', side_effect=evaluate):
                    result = experiment_syn.run_subject(self.augmented, torch.device('cpu'),
                        replace(self.config, num_epochs=3, combine_valid_test=True))
                self.assertEqual(result['selected_epoch'], 1 if zero_scores else 2)
                self.assertEqual(float(model.weight.detach().item()), result['selected_epoch'])
                self.assertEqual(result['test']['confusion_matrix'], [[0, 2], [2, 0]] if zero_scores else [[2, 0], [1, 1]])
                self.assertEqual(evaluated, ['training', 'combined'] * 3)
                self.assertEqual(result['evaluation_trial_count'], 4)
                self.assertNotIn('best_validation_accuracy', result)

    def test_combined_cli_marks_selection_protocol_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'combined.json'
            with redirect_stdout(io.StringIO()):
                main(['--original-dir', str(self.original), '--synthetic-dir', str(self.synthetic),
                      '--subjects', '1', '--epochs', '2', '--device', 'cpu', '--quiet',
                      '--combine-valid-test', '--output', str(path)])
            result = json.loads(path.read_text())
            self.assertEqual(result['selection_policy'], 'best_combined_test_accuracy')
            self.assertFalse(result['independent_final_test'])
            self.assertEqual(result['test_evaluations_per_subject'], 2)
            self.assertEqual(result['evaluation_partitions'], ['validation', 'testing'])
            self.assertEqual(result['subjects']['1']['counts']['training']['total'], 12)
            self.assertEqual(sum(map(sum, result['summary']['confusion_matrix'])), 4)

    def test_cli_both_conditions_reuse_learning_defaults_and_save_only_results(self):
        standard = default_training_config()
        actual = SynTrainingConfig().standard()
        for name in ('batch_size', 'learning_rate', 'num_epochs', 'hidden_channels', 'heads', 'fold_seed', 'optimizer', 'loss'):
            self.assertEqual(getattr(actual, name), getattr(standard, name))
        with tempfile.TemporaryDirectory() as directory:
            for source, count, condition in ((None, 8, 'original_only'), (self.synthetic, 12, 'original_plus_synthetic')):
                path = Path(directory) / f'{condition}.json'
                argv = ['--original-dir', str(self.original), '--subjects', '1', '--epochs', '1',
                        '--device', 'cpu', '--batch-size', '4', '--quiet', '--output', str(path)]
                if source:
                    argv += ['--synthetic-dir', str(source)]
                with redirect_stdout(io.StringIO()):
                    main(argv)
                result = json.loads(path.read_text())
                self.assertEqual(result['condition'], condition)
                self.assertEqual(result['selection_policy'], 'best_validation_accuracy')
                self.assertEqual(result['subjects']['1']['counts']['training']['total'], count)
                self.assertEqual(result['subjects']['1']['optimizer_updates'], count // 4)
                self.assertEqual(sum(map(sum, result['summary']['confusion_matrix'])), 2)
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_original_kfold_membership_and_counts(self):
        config = replace(self.config, protocol='kfold', n_folds=10)
        original = [Data(y=torch.tensor(i % 2), subject_id=1, session_id=3,
                         trial_id=i, sample_id=f'original:{i}', is_synthetic=False,
                         partition='training' if i < 100 else 'validation' if i < 122 else 'testing')
                    for i in range(144)]
        synthetic = [Data(y=torch.tensor(i % 2), subject_id=1, session_id=-1,
                          trial_id=i, sample_id=f'synthetic:{i}', is_synthetic=True, partition='training')
                     for i in range(100)]
        partitions = {name: [g for g in original + synthetic if g.partition == name]
                      for name in ('training', 'validation', 'testing')}
        folds = list(make_augmented_folds(partitions, config))
        seen = []
        for training, testing in folds:
            self.assertIn(len(training), (229, 230))
            self.assertIn(len(testing), (14, 15))
            self.assertEqual(sum(g.is_synthetic for g in training), 100)
            self.assertFalse(any(g.is_synthetic for g in testing))
            self.assertFalse({g.sample_id for g in training} & {g.sample_id for g in testing})
            seen.extend(g.sample_id for g in testing)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {g.sample_id for g in original})
        reversed_partitions = {k: list(reversed(v)) for k, v in partitions.items()}
        self.assertEqual([[g.sample_id for g in test] for _, test in folds],
                         [[g.sample_id for g in test] for _, test in make_augmented_folds(reversed_partitions, config)])
        baseline = {k: [g for g in v if not g.is_synthetic] for k, v in partitions.items()}
        self.assertEqual([[g.sample_id for g in test] for _, test in folds],
                         [[g.sample_id for g in test] for _, test in make_augmented_folds(baseline, config)])

    def test_kfold_cli_reuses_standard_trainer_and_reports_all_originals(self):
        from train.experiment import run_fold
        self.assertIs(experiment_syn.run_fold, run_fold)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'folds.json'
            with redirect_stdout(io.StringIO()):
                main(['--original-dir', str(self.original), '--synthetic-dir', str(self.synthetic),
                      '--subjects', '1', '--epochs', '1', '--batch-size', '8', '--device', 'cpu',
                      '--quiet', '--protocol', 'kfold', '--folds', '3', '--output', str(path)])
            result = json.loads(path.read_text())
            self.assertEqual(result['selection_policy'], 'best_test_accuracy')
            self.assertFalse(result['independent_final_test'])
            self.assertTrue(result['synthetic_reused_across_folds'])
            folds = result['subjects']['1']['folds']
            self.assertEqual(len(folds), 3)
            for fold in folds:
                self.assertEqual(fold['counts']['training']['total'], 12)
                self.assertEqual(fold['counts']['training']['synthetic'], 4)
                self.assertEqual(fold['counts']['testing']['total'], 4)
                self.assertEqual(fold['selected_epoch'], 1)
                self.assertEqual(fold['optimizer_updates'], 2)
            self.assertEqual(sum(map(sum, result['summary']['confusion_matrix'])), 12)
        with self.assertRaisesRegex(ValueError, 'combine-valid-test'):
            replace(self.config, protocol='kfold', combine_valid_test=True).validate()

    def test_standalone_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'copied'
            for name in ('dataset', 'model', 'train', 'datacustom'):
                shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns('__pycache__'))
            output = Path(directory) / 'results.json'
            env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            result = subprocess.run([sys.executable, str(target / 'train/train_syn.py'),
                '--original-dir', str(self.original), '--synthetic-dir', str(self.synthetic),
                '--subjects', '1', '--epochs', '1', '--device', 'cpu', '--quiet', '--output', str(output)],
                cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())['condition'], 'original_plus_synthetic')


if __name__ == '__main__':
    unittest.main()
