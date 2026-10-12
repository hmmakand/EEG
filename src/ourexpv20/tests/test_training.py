"""Training behavior checks against the original trainer and known outcomes."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import build_subject_dataset, default_dataset_config, legacy_dataset_config
from fixture_data import write_subject
from model.gat import GAT
from model.custom import GritGPSModel
from train import experiment
from train.config import default_training_config, validate_training_config
from train.engine import evaluate, Evaluation
from train.metrics import summarize_fold_scores, compute_classification_metrics
from train.reporting import save_results
from train.selection import initialize_selection, update_selection, finalize_selection
from train.setup import seed_fold, build_model
from train.splitting import make_fold_indices
from train import train as cli


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        write_subject(cls.temp.name)
        cls.dataset_config = default_dataset_config()
        cls.training_config = replace(default_training_config(), subjects=(1,), n_folds=2,
                                      num_epochs=3, show_progress=False, device='cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_original_training_equivalence(self):
        # The original trainer: GATv2, unshuffled batches, no weight decay.
        training_config = replace(self.training_config, model='gat', train_shuffle=False, weight_decay=0.)
        expected = json.loads((Path(__file__).parent / 'fixtures/legacy_training.json').read_text())
        legacy_config = legacy_dataset_config()
        splits = [{'train': a.tolist(), 'test': b.tolist()}
                  for a, b in make_fold_indices(list(range(12)), training_config)]
        self.assertEqual(splits, expected['folds'])
        observed = []
        original_evaluate = experiment.evaluate
        def capture(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            observed.append(result.accuracy if isinstance(result, Evaluation) else result)
            return result
        with patch.object(experiment, 'evaluate', side_effect=capture):
            actual = experiment.run_subject(self.temp.name, 1, torch.device('cpu'),
                                              legacy_config, training_config)
        np.testing.assert_allclose(observed, [score for epoch in expected['epochs']
                                             for score in (epoch['train'], epoch['test'])], rtol=0, atol=0)
        self.assertEqual({key: actual[key] for key in ('mean', 'max', 'min')}, expected['summary'])
        self.assertEqual(sum(map(sum, actual['confusion_matrix'])), 12)
        self.assertEqual(len(actual['folds']), 2)

    def test_seed_and_initial_model_match_original(self):
        torch.manual_seed(12345)
        original = GAT(hidden_channels=22, heads=3, in_channels=8)
        seed_fold(self.training_config.fold_seed)
        actual = build_model(8, torch.device('cpu'), replace(self.training_config, model='gat'))
        for key, tensor in original.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], tensor, rtol=0, atol=0)

    def test_default_model_is_custom(self):
        self.assertEqual(default_training_config().model, 'custom')
        model = build_model(8, torch.device('cpu'), self.training_config, num_nodes=22)
        self.assertIsInstance(model, GritGPSModel)
        self.assertEqual(model.cfg.num_electrodes, 22)
        # Legacy graphs carry (0, 0) placeholders; a batch may hold no real edge.
        x, placeholders = torch.randn(44, 8), torch.zeros((2, 968), dtype=torch.long)
        batch = torch.arange(2).repeat_interleave(22)
        for mode in (model.train, model.eval):
            mode()
            out = model(x, placeholders, batch)
            self.assertEqual(tuple(out.shape), (2, 2))
            self.assertTrue(torch.isfinite(out).all())
        with self.assertRaises(ValueError):
            validate_training_config(replace(self.training_config, model='unknown'))

    def test_selection_preserves_strict_improvement(self):
        state = initialize_selection()
        for epoch, (train_accuracy, test_accuracy) in enumerate([(1., .5), (.8, .75), (.9, .75), (1., .25)], start=1):
            state = update_selection(state, epoch, train_accuracy, test_accuracy)
        self.assertEqual(state.epoch, 2)
        self.assertEqual(state.train_accuracy, .8)
        self.assertEqual(finalize_selection(state), .75)
        self.assertEqual(finalize_selection(update_selection(initialize_selection(), 1, 1., 0.)), 0.)

    def test_existing_metrics_and_json(self):
        summary = summarize_fold_scores([.25, .5, .75])
        self.assertEqual(summary, {'mean': .5, 'max': .75, 'min': .25})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested/results.json'
            with redirect_stdout(io.StringIO()):
                save_results({1: summary}, path)
            self.assertEqual(json.loads(path.read_text()), {'1': summary})
        with self.assertRaises(ValueError):
            summarize_fold_scores([])

    def test_accuracy_weights_samples_not_batches(self):
        class FixedModel(torch.nn.Module):
            def forward(self, x, edge_index, batch):
                return x  # Each graph has exactly one node with known logits.
        graphs = [Data(x=torch.tensor([[2., 0.]]), y=torch.tensor(label),
                       edge_index=torch.empty((2, 0), dtype=torch.long)) for label in (0, 0, 1)]
        model = FixedModel()
        accuracy = evaluate(model, DataLoader(graphs, batch_size=2), torch.device('cpu'))
        self.assertEqual(accuracy, 2 / 3)
        self.assertFalse(model.training)

    def test_temporal_encoder_reads_node_signals(self):
        from train.engine import predict
        from train.setup import sample_dimensions
        graphs = build_subject_dataset(self.temp.name, 1, default_dataset_config())[:4]
        dimensions = sample_dimensions(graphs[0])
        self.assertEqual(dimensions, {'num_nodes': 22, 'edge_dim': 2, 'signal_samples': 500})
        model = build_model(graphs[0].num_node_features, torch.device('cpu'), self.training_config, **dimensions)
        self.assertEqual(model.cfg.temporal, 'head')
        self.assertTrue(model.uses_signal)
        batch = next(iter(DataLoader(graphs, batch_size=4)))
        out = predict(model, batch)
        self.assertEqual(tuple(out.shape), (4, 2))
        out.sum().backward()
        self.assertGreater(float(model.temporal.conv.weight.grad.abs().sum()), 0)
        changed = batch.clone()
        changed.signal = changed.signal.flip(-1) * 2
        model.eval()
        self.assertFalse(torch.allclose(predict(model, batch), predict(model, changed)))
        with self.assertRaises(ValueError):
            model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        # Without node signals the model is built without the temporal encoder.
        plain = build_model(8, torch.device('cpu'), self.training_config, num_nodes=22)
        self.assertEqual(plain.cfg.temporal, 'none')

    def test_center_logits_removes_a_constant_class_bias(self):
        class BiasedModel(torch.nn.Module):
            def forward(self, x, edge_index, batch):
                return x + torch.tensor([5., 0.])  # always favours class 0
        graphs = [Data(x=torch.tensor([[0., 1.]] if label else [[1., 0.]]), y=torch.tensor(label),
                       edge_index=torch.empty((2, 0), dtype=torch.long)) for label in (0, 1, 0, 1)]
        loader = DataLoader(graphs, batch_size=3)
        self.assertEqual(evaluate(BiasedModel(), loader, torch.device('cpu')), .5)
        centred = evaluate(BiasedModel(), loader, torch.device('cpu'), center_logits=True, collect_predictions=True)
        self.assertEqual(centred.accuracy, 1.)
        self.assertEqual(centred.predictions, [0, 1, 0, 1])

    def test_cli_runs_experiment_and_saves_original_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            config = replace(self.training_config, num_epochs=1)
            with patch.object(cli, 'default_training_config', return_value=config), redirect_stdout(io.StringIO()):
                cli.main(['--data-dir', self.temp.name, '--output', str(path)])
            results = json.loads(path.read_text())
            self.assertEqual(set(results), {'1'})
            self.assertEqual(set(results['1']), {'mean', 'max', 'min', 'balanced_accuracy', 'macro_f1', 'confusion_matrix', 'class_labels', 'folds'})
            self.assertEqual(results['1']['class_labels'], [0, 1])
            self.assertEqual(sum(map(sum, results['1']['confusion_matrix'])), 12)

    def test_classification_metrics_known_predictions(self):
        metrics = compute_classification_metrics([0, 0, 0, 1], [0, 0, 1, 1])
        self.assertAlmostEqual(metrics['balanced_accuracy'], 5 / 6)
        self.assertAlmostEqual(metrics['macro_f1'], 11 / 15)
        self.assertEqual(metrics['confusion_matrix'], [[2, 1], [0, 1]])
        absent = compute_classification_metrics([0, 0], [0, 0])
        self.assertEqual(absent['confusion_matrix'], [[2, 0], [0, 0]])
        self.assertEqual(absent['balanced_accuracy'], 1.)
        self.assertEqual(absent['macro_f1'], .5)
        with self.assertRaises(ValueError):
            compute_classification_metrics([], [])

    def test_metrics_use_selected_epoch_and_keep_first_tie(self):
        graphs = [Data(x=torch.zeros((1, 8)), y=torch.tensor(label),
                       edge_index=torch.empty((2, 0), dtype=torch.long)) for label in (0, 0, 0, 1)]
        evaluations = [
            .5, Evaluation(.5, [0, 0, 0, 1], [0, 1, 1, 1]),
            .8, Evaluation(.75, [0, 0, 0, 1], [0, 0, 1, 1]),
            .9, Evaluation(.75, [0, 0, 0, 1], [0, 0, 0, 0]),
        ]
        with patch.object(experiment, 'train_epoch'), patch.object(experiment, 'evaluate', side_effect=evaluations) as evaluated:
            result = experiment.run_fold(graphs, graphs, torch.device('cpu'), self.training_config, 1, 1)
        self.assertEqual(evaluated.call_count, 6)
        self.assertEqual(result['selected_epoch'], 2)
        self.assertEqual(result['accuracy'], .75)
        self.assertEqual(result['confusion_matrix'], [[2, 1], [0, 1]])
        self.assertAlmostEqual(result['macro_f1'], 11 / 15)
        self.assertAlmostEqual(result['balanced_accuracy'], 5 / 6)

    def test_all_zero_accuracy_still_has_selected_metrics(self):
        graph = Data(x=torch.zeros((1, 8)), y=torch.tensor(0), edge_index=torch.empty((2, 0), dtype=torch.long))
        with patch.object(experiment, 'train_epoch'), patch.object(experiment, 'evaluate',
                side_effect=[0., Evaluation(0., [0, 1], [1, 0])]):
            result = experiment.run_fold([graph], [graph], torch.device('cpu'),
                                         replace(self.training_config, num_epochs=1), 1, 1)
        self.assertEqual(result['selected_epoch'], 1)
        self.assertEqual(result['accuracy'], 0.)
        self.assertEqual(result['balanced_accuracy'], 0.)
        self.assertEqual(result['macro_f1'], 0.)
        self.assertEqual(result['confusion_matrix'], [[0, 1], [1, 0]])

    def test_invalid_settings_and_insufficient_data(self):
        for config in (replace(self.training_config, num_epochs=0),
                       replace(self.training_config, n_folds=1),
                       replace(self.training_config, learning_rate=-1),
                       replace(self.training_config, selection_policy='new_metric')):
            with self.assertRaises(ValueError):
                validate_training_config(config)
        with self.assertRaisesRegex(ValueError, 'at least 2 graphs'):
            make_fold_indices([0], self.training_config)
        with self.assertRaisesRegex(ValueError, 'Missing data files'):
            cli.validate_input_files(self.temp.name, (2,))
        self.assertEqual(default_training_config().num_epochs, 249)
        self.assertEqual(default_dataset_config().graphs.threshold, .35)

    def test_repository_module_entry_point(self):
        result = subprocess.run([sys.executable, '-m', f'{ROOT.name}.train.train', '--help'],
                                cwd=ROOT.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--data-dir', result.stdout)


if __name__ == '__main__':
    unittest.main()
