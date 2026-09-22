"""FixMatch pseudo-labeling, mask thresholding, and fold integration."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.config import TrainingConfig, validate_training_config
from train.experiment import run_fold
from train.fixmatch_engine import train_epoch_fixmatch, validate_fixmatch_settings


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x, edge_index, batch):
        return self.linear(global_mean_pool(x, batch))


def graphs(labeled=True):
    result = []
    for index, nodes in enumerate((2, 4, 3)):
        graph = Data(x=torch.randn(nodes, 3), edge_index=torch.empty((2, 0), dtype=torch.long))
        if labeled:
            graph.y = torch.tensor(index % 2)
        result.append(graph)
    return result


class FixMatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def test_invalid_settings(self):
        for kwargs in ({'lambda_u': -1}, {'confidence_threshold': 0}, {'confidence_threshold': 1.5},
                       {'weak_noise_std': float('nan')}, {'strong_mask_prob': 1.0}, {'strong_mask_prob': -0.1}):
            base = dict(lambda_u=1.0, confidence_threshold=.95, weak_noise_std=.05,
                       strong_noise_std=.2, strong_mask_prob=.3)
            base.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_fixmatch_settings(**base)

    def test_empty_labeled_loader_raises(self):
        model = ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with self.assertRaisesRegex(ValueError, 'empty labeled'):
            train_epoch_fixmatch(model, [], optimizer, 'cpu')

    def test_zero_lambda_u_skips_pseudo_labeling(self):
        model = ToyModel()
        labeled = DataLoader(graphs(), batch_size=2)
        unlabeled = DataLoader(graphs(False), batch_size=2)
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            metrics = train_epoch_fixmatch(
                model, labeled, torch.optim.Adam(model.parameters()), 'cpu',
                unlabeled_loader=unlabeled, lambda_u=0.)
        # Only the labeled forward passes should run; no weak/strong unlabeled forwards.
        self.assertEqual(forward.call_count, len(labeled))
        self.assertEqual(metrics.unsupervised_loss, 0.)
        self.assertEqual(metrics.pseudo_label_mask_rate, 0.)

    def test_low_confidence_threshold_produces_gradient_from_unlabeled_data(self):
        model = ToyModel()
        before = deepcopy(model.state_dict())
        labeled = DataLoader(graphs(), batch_size=3)
        unlabeled = DataLoader(graphs(False), batch_size=3)
        metrics = train_epoch_fixmatch(
            model, labeled, torch.optim.Adam(model.parameters()), 'cpu',
            unlabeled_loader=unlabeled, lambda_u=1.0, confidence_threshold=1e-6,
            weak_noise_std=0., strong_noise_std=.1, strong_mask_prob=0.)
        self.assertAlmostEqual(metrics.pseudo_label_mask_rate, 1.0, places=5)
        self.assertGreater(metrics.unsupervised_loss, 0.)
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.state_dict().items()))
        self.assertTrue(model.training)

    def test_high_confidence_threshold_masks_everything(self):
        model = ToyModel()
        labeled = DataLoader(graphs(), batch_size=3)
        unlabeled = DataLoader(graphs(False), batch_size=3)
        metrics = train_epoch_fixmatch(
            model, labeled, torch.optim.Adam(model.parameters()), 'cpu',
            unlabeled_loader=unlabeled, lambda_u=1.0, confidence_threshold=1.0 - 1e-9)
        self.assertEqual(metrics.pseudo_label_mask_rate, 0.)
        self.assertEqual(metrics.unsupervised_loss, 0.)

    def test_metrics_finite_and_total_matches_sum(self):
        model = ToyModel()
        labeled = DataLoader(graphs(), batch_size=1)
        unlabeled = DataLoader(graphs(False)[:1], batch_size=1)
        metrics = train_epoch_fixmatch(model, labeled, torch.optim.Adam(model.parameters()), 'cpu',
                                       unlabeled_loader=unlabeled, confidence_threshold=1e-6)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in vars(metrics).values()))
        self.assertAlmostEqual(metrics.total_loss, metrics.supervised_loss + metrics.unsupervised_loss, places=6)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            validate_training_config(replace(
                TrainingConfig(training_engine='fixmatch'), fixmatch_confidence_threshold=0.))
        validate_training_config(TrainingConfig(training_engine='fixmatch'))

    def test_fold_integration_without_unlabeled_loader(self):
        config = TrainingConfig(training_engine='fixmatch', num_epochs=1, hidden_channels=3,
                                heads=1, show_progress=False)
        with patch('train.experiment.train_epoch_fixmatch', wraps=train_epoch_fixmatch) as trained:
            result = run_fold(graphs(), graphs(), torch.device('cpu'), config, 1, 1)
        trained.assert_called_once()
        self.assertIsNone(trained.call_args.kwargs['unlabeled_loader'])
        self.assertTrue(0 <= result['accuracy'] <= 1)

    def test_fold_integration_with_unlabeled_loader(self):
        config = TrainingConfig(training_engine='fixmatch', num_epochs=1, hidden_channels=3,
                                heads=1, show_progress=False, fixmatch_confidence_threshold=1e-6)
        with patch('train.experiment.train_epoch_fixmatch', wraps=train_epoch_fixmatch) as trained:
            result = run_fold(graphs(), graphs(), torch.device('cpu'), config, 1, 1,
                              unlabeled_loader=DataLoader(graphs(False), batch_size=2))
        trained.assert_called_once()
        self.assertIsNotNone(trained.call_args.kwargs['unlabeled_loader'])
        self.assertTrue(0 <= result['accuracy'] <= 1)

    def test_unlabeled_loader_rejected_for_supervised_engine(self):
        config = TrainingConfig(training_engine='supervised', num_epochs=1, hidden_channels=3, heads=1)
        with self.assertRaisesRegex(ValueError, "requires training_engine"):
            run_fold(graphs(), graphs(), torch.device('cpu'), config, 1, 1,
                    unlabeled_loader=DataLoader(graphs(False), batch_size=2))

    def test_reuse_labeled_as_unlabeled_config_validation(self):
        with self.assertRaisesRegex(ValueError, 'fixmatch_reuse_labeled_as_unlabeled requires'):
            validate_training_config(TrainingConfig(
                training_engine='supervised', fixmatch_reuse_labeled_as_unlabeled=True))
        validate_training_config(TrainingConfig(
            training_engine='fixmatch', fixmatch_reuse_labeled_as_unlabeled=True))

    def test_fold_builds_unlabeled_loader_from_train_data_when_reuse_enabled(self):
        config = TrainingConfig(training_engine='fixmatch', num_epochs=1, hidden_channels=3,
                                heads=1, show_progress=False, fixmatch_confidence_threshold=1e-6,
                                fixmatch_reuse_labeled_as_unlabeled=True)
        train_data = graphs()
        with patch('train.experiment.train_epoch_fixmatch', wraps=train_epoch_fixmatch) as trained:
            result = run_fold(train_data, graphs(), torch.device('cpu'), config, 1, 1)
        trained.assert_called_once()
        built_loader = trained.call_args.kwargs['unlabeled_loader']
        self.assertIsNotNone(built_loader)
        self.assertEqual(sum(batch.num_graphs for batch in built_loader), len(train_data))
        self.assertTrue(0 <= result['accuracy'] <= 1)

    def test_reuse_labeled_as_unlabeled_rejects_explicit_unlabeled_loader(self):
        config = TrainingConfig(training_engine='fixmatch', num_epochs=1, hidden_channels=3,
                                heads=1, show_progress=False, fixmatch_reuse_labeled_as_unlabeled=True)
        with self.assertRaisesRegex(ValueError, 'not both'):
            run_fold(graphs(), graphs(), torch.device('cpu'), config, 1, 1,
                    unlabeled_loader=DataLoader(graphs(False), batch_size=2))


if __name__ == '__main__':
    unittest.main()
