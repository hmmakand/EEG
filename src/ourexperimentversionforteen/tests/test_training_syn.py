"""Saved-fold membership, graph equivalence, provenance, and supervised reporting."""
from contextlib import redirect_stdout
from dataclasses import replace
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixture_data import write_subject
from dataset import build_trial_graph, default_dataset_config
from dataset.trials import Trial
from datasetsynthetic.config import PreparationConfig
from datasetsynthetic.prepare import prepare
from datasetsynthetic.storage import read_rows, write_rows, write_json, sha256
from datacustom import SavedWithinSubject
from datacustom import dataset as adapter
from train.config_syn import SynTrainingConfig
from train import experiment_syn, engine
from train.experiment import run_fold as standard_run_fold
from train.train_syn import main


def complete_fixture(root):
    """Small synthetic fixture with real preparation artifacts; no diffusion training."""
    write_subject(root)
    shutil.copy2(root / 'A01T.mat', root / 'A02T.mat')
    bundle = root / 'bundle'
    with redirect_stdout(io.StringIO()):
        prepare(root, bundle, PreparationConfig(subjects=(1, 2), n_folds=2,
                                                samples_per_class=2, validation_per_class=1))
    for job in json.loads((bundle / 'jobs.json').read_text()):
        if job['protocol'] != 'within_subject':
            continue
        subject = job['subject_id']
        run = bundle / job['run_directory']
        run.mkdir(parents=True)
        inputs = bundle / job['input_directory']
        permitted = read_rows(inputs / 'partitions.csv')
        val_ids = {next(r['trial_id'] for r in permitted if r['class_label'] == str(c)) for c in (0, 1)}
        internal = [dict(r, diffusion_subset='validation' if r['trial_id'] in val_ids else 'training') for r in permitted]
        write_rows(run / 'internal_partitions.csv', internal)
        config = dict(subjects=[subject], samples_per_class=2, validation_per_class=1, epochs=1)
        write_json(run / 'config.json', config)
        name = f'subject_{subject:02d}.npz'
        with np.load(bundle / 'original' / name) as data:
            signals = np.stack([data['signals'][int(next(r['array_index'] for r in permitted if r['class_label'] == str(c)))] * .9
                                for c in (0, 0, 1, 1)])
        ids = np.array([f'S{subject:02d}_synthetic_{i:05d}' for i in range(4)])
        np.savez_compressed(run / name, signals=signals, labels=np.array([0, 0, 1, 1]),
                            subject_ids=np.full(4, subject), synthetic_trial_ids=ids)
        write_rows(run / 'manifest.csv', [dict(subject_id=subject, array_index=i, class_label=i // 2,
                    output_file=name, synthetic_trial_id=ids[i], checkpoint_sha256='fixture',
                    source='synthetic', intended_partition='training') for i in range(4)])
        write_json(run / 'metadata.json', dict(status='complete',
            subjects={str(subject): {'epoch': 1}},
            generation={str(subject): dict(sha256=sha256(run / name), checkpoint_sha256='fixture')},
            original_sha256={**{n: sha256(inputs / n) for n in ('metadata.json', 'partitions.csv')},
                             name: sha256(bundle / 'original' / name)},
            manifest_sha256=sha256(run / 'manifest.csv'),
            internal_partitions_sha256=sha256(run / 'internal_partitions.csv')))
    final = bundle / 'protocols/within_subject.csv'
    shutil.copy2(bundle / 'protocols/within_subject.plan.csv', final)
    write_json(bundle / 'generation_status.json', dict(complete=False, protocols={
        'within_subject': dict(complete=True, manifest_sha256=sha256(final)),
        'cross_subject': dict(complete=False)}))
    return bundle


def rewrite_manifest(bundle, rows):
    """Update hashes as well to test structural validation beyond checksum checks."""
    for name in ('within_subject.csv', 'within_subject.plan.csv'):
        write_rows(bundle / 'protocols' / name, rows)
    metadata = json.loads((bundle / 'metadata.json').read_text())
    metadata['artifact_sha256']['protocols/within_subject.plan.csv'] = sha256(bundle / 'protocols/within_subject.plan.csv')
    write_json(bundle / 'metadata.json', metadata)
    status = json.loads((bundle / 'generation_status.json').read_text())
    status['protocols']['within_subject']['manifest_sha256'] = sha256(bundle / 'protocols/within_subject.csv')
    write_json(bundle / 'generation_status.json', status)


class SavedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.bundle = complete_fixture(cls.root)
        cls.config = SynTrainingConfig(subjects=(1,), num_epochs=1, hidden_channels=3,
                                      heads=1, device='cpu', show_progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_graph_equivalence_cache_and_exact_saved_membership(self):
        with patch.object(adapter, 'build_trial_graph', wraps=build_trial_graph) as builder:
            data = SavedWithinSubject(self.bundle)
            first = data.load_fold(1, 1)
            self.assertEqual(builder.call_count, 12 + 4)
            second = data.load_fold(1, 2)
            self.assertEqual(builder.call_count, 12 + 8)
        self.assertEqual((len(first.training), len(first.testing)), (10, 6))
        self.assertEqual(data.subjects, (1, 2))
        originals = {g.sample_id: g for g in first.training + first.testing if not g.is_synthetic}
        for g in second.training + second.testing:
            if not g.is_synthetic:
                self.assertIs(g, originals[g.sample_id])
        for fold in (first, second):
            for role in ('training', 'testing'):
                self.assertEqual([g.sample_id for g in getattr(fold, role)],
                    [r['trial_id'] for r in data.assignments[(1, fold.fold)] if r['role'] == role])
            self.assertFalse({g.sample_id for g in fold.training} & {g.sample_id for g in fold.testing})
            self.assertFalse(any(g.is_synthetic for g in fold.testing))
            train, test = fold.loaders(8)
            batch = next(iter(train))
            self.assertEqual(tuple(batch.x.shape), (8 * 22, 8))
            self.assertEqual(test.batch_size, 8)
        self.assertEqual(len({g.sample_id for f in (first, second) for g in f.testing}), 12)
        for synthetic in (False, True):
            graph = next(g for g in first.training if bool(g.is_synthetic) == synthetic)
            row = next(r for r in data.assignments[(1, 1)] if r['trial_id'] == graph.sample_id)
            with np.load(self.bundle / row['signal_file']) as arrays:
                direct = build_trial_graph(Trial(arrays['signals'][int(row['array_index'])], int(row['class_label']),
                                                1, int(graph.session_id), int(graph.trial_id)))
            torch.testing.assert_close(graph.x, direct.x, rtol=0, atol=0)
            torch.testing.assert_close(graph.edge_index, direct.edge_index, rtol=0, atol=0)

    def test_rejects_incomplete_protocol_and_bad_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'bundle'
            shutil.copytree(self.bundle, bundle)
            status = json.loads((bundle / 'generation_status.json').read_text())
            status['protocols']['within_subject']['complete'] = False
            write_json(bundle / 'generation_status.json', status)
            with self.assertRaisesRegex(ValueError, 'not complete'):
                SavedWithinSubject(bundle)
            shutil.copy2(self.bundle / 'generation_status.json', bundle / 'generation_status.json')
            with (bundle / 'protocols/within_subject.csv').open('a') as stream:
                stream.write('changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                SavedWithinSubject(bundle)

    def test_rejects_overlap_wrong_fold_and_synthetic_testing(self):
        for change in ('overlap', 'wrong_fold', 'synthetic_test'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                bundle = Path(directory) / 'bundle'
                shutil.copytree(self.bundle, bundle)
                rows = read_rows(bundle / 'protocols/within_subject.csv')
                if change == 'overlap':
                    rows.append(dict(rows[0], role='testing'))
                else:
                    r = next(r for r in rows if r['source'] == 'synthetic')
                    if change == 'wrong_fold':
                        r['generator_job'] = 'within_subject/S01/fold_02'
                    else:
                        r['role'] = 'testing'
                rewrite_manifest(bundle, rows)
                with self.assertRaises(ValueError):
                    SavedWithinSubject(bundle)

    def test_rejects_generated_signal_and_generator_input_changes(self):
        for change in ('signals', 'inputs'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                bundle = Path(directory) / 'bundle'
                shutil.copytree(self.bundle, bundle)
                run = bundle / 'synthetic/within_subject/S01/fold_01'
                if change == 'signals':
                    with (run / 'subject_01.npz').open('ab') as stream:
                        stream.write(b'changed')
                else:
                    meta = json.loads((run / 'metadata.json').read_text())
                    meta['original_sha256']['partitions.csv'] = 'another_fold'
                    write_json(run / 'metadata.json', meta)
                with self.assertRaises(ValueError):
                    SavedWithinSubject(bundle).load_fold(1, 1)

    def test_training_matches_standard_and_never_splits(self):
        fold = SavedWithinSubject(self.bundle).load_fold(1, 1)
        self.assertIs(experiment_syn.train_epoch, engine.train_epoch)
        self.assertIs(experiment_syn.evaluate, engine.evaluate)
        standard = standard_run_fold(fold.training, fold.testing, torch.device('cpu'), self.config.standard(), 1, 1)
        with patch('train.splitting.make_fold_indices', side_effect=AssertionError('Must not split')):
            result = experiment_syn.run_fold(fold, torch.device('cpu'), self.config)
        for key, value in standard.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result['optimizer_updates'], 2)

    def test_selected_epoch_metrics_keep_first_tie_and_all_zero(self):
        fold = SavedWithinSubject(self.bundle).load_fold(1, 1)
        for scores, expected in [([.5, .75, .75], 2), ([0., 0., 0.], 1)]:
            predictions = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1]] if expected == 2 else [[1, 0, 1, 0]] * 3
            side_effect = []
            for score, pred in zip(scores, predictions):
                side_effect.extend([.5, engine.Evaluation(score, [0, 1, 0, 1], pred)])
            with patch.object(experiment_syn, 'train_epoch') as train, \
                 patch.object(experiment_syn, 'evaluate', side_effect=side_effect) as evaluate:
                result = experiment_syn.run_fold(fold, torch.device('cpu'), replace(self.config, num_epochs=3))
            self.assertEqual(train.call_count, 3)
            self.assertEqual(evaluate.call_count, 6)
            self.assertEqual(result['selected_epoch'], expected)
            self.assertEqual(result['confusion_matrix'], [[2, 0], [1, 1]] if expected == 2 else [[0, 2], [2, 0]])

    def test_cli_reports_saved_folds_and_rejects_removed_options(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            with redirect_stdout(io.StringIO()):
                main(['--bundle', str(self.bundle), '--subjects', '1', '--epochs', '1',
                      '--device', 'cpu', '--quiet', '--output', str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(result['saved_fold_counts'], {'1': 2})
            self.assertEqual(result['training_config']['batch_size'], 8)
            self.assertNotIn('n_folds', result['training_config'])
            self.assertEqual(result['selection_policy'], 'best_test_accuracy')
            self.assertFalse(result['independent_final_test'])
            self.assertEqual(sum(map(sum, result['summary']['confusion_matrix'])), 12)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
        for option in ('--folds', '--combine-valid-test', '--original-dir', '--synthetic-dir', '--protocol'):
            with self.subTest(option=option), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                main([option, '2'])
        with self.assertRaisesRegex(ValueError, 'Requested subjects'):
            experiment_syn.run_experiment(self.bundle, torch.device('cpu'), default_dataset_config(), replace(self.config, subjects=(9,)))

    def test_standalone_cli_without_diffusion_code(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'copied'
            for name in ('dataset', 'model', 'train', 'datacustom'):
                shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns('__pycache__'))
            (target / 'datasetsynthetic').mkdir()
            for name in ('__init__.py', 'storage.py'):
                shutil.copy2(ROOT / 'datasetsynthetic' / name, target / 'datasetsynthetic' / name)
            bundle = target / 'datasyn/fold_first'
            shutil.copytree(self.bundle, bundle)
            env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            result = subprocess.run([sys.executable, str(target / 'train/train_syn.py'),
                '--subjects', '1', '--epochs', '1', '--device', 'cpu', '--quiet'],
                cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output, = (target / 'output').glob('*.json')
            self.assertEqual(json.loads(output.read_text())['executed_subjects'], [1])


if __name__ == '__main__':
    unittest.main()
