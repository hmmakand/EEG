"""Ensure LOSO never uses held-out performance for epoch selection."""
import unittest
from unittest.mock import patch, MagicMock

import torch
from torch_geometric.data import Data

from src.ourexpv19.datacustom.dataset import FoldGraphs
from src.ourexpv19.train.config_syn import SynTrainingConfig
from src.ourexpv19.train import experiment_syn_loso as runner
from src.ourexpv19.train.engine import Evaluation


class FinalEpochTests(unittest.TestCase):
    def test_test_subject_evaluated_once_after_all_epochs(self):
        graph = Data(x=torch.ones(2, 8), edge_index=torch.tensor([[0, 1], [1, 0]]),
                     y=torch.tensor([0]), sample_id='trial', is_synthetic=False)
        other = graph.clone()
        other.y = torch.tensor([1])
        other.sample_id = 'other'
        fold = FoldGraphs(1, 1, [graph, other], [graph, other], {})
        events = []
        with patch.object(runner, 'build_model', return_value=MagicMock()), \
             patch.object(runner, 'build_optimizer'), patch.object(runner, 'build_criterion'), \
             patch.object(runner, 'train_epoch', side_effect=lambda *a: events.append('train')), \
             patch.object(runner, 'evaluate', side_effect=lambda *a, **kw: (events.append('test') or Evaluation(1.0, [0, 1], [0, 1]))) as evaluation:
            result = runner.run_fold(fold, torch.device('cpu'),
                                     SynTrainingConfig(num_epochs=3, show_progress=False))
        self.assertEqual(events, ['train', 'train', 'train', 'test'])
        self.assertEqual(evaluation.call_count, 1)
        self.assertTrue(evaluation.call_args.kwargs['collect_predictions'])
        self.assertEqual(result['selected_epoch'], 3)
        self.assertEqual(result['optimizer_updates'], 3)
        self.assertFalse(evaluation.call_args.kwargs['center_logits'])

    def test_ensemble_trains_every_member_and_averages_probabilities(self):
        from src.ourexpv19.train.setup import Ensemble

        class Constant(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = torch.tensor(logits)

            def forward(self, x, edge_index, batch):
                return self.logits.expand(int(batch.max()) + 1, 2)

        graph = Data(x=torch.ones(2, 8), edge_index=torch.tensor([[0, 1], [1, 0]]),
                     y=torch.tensor([0]), sample_id='trial', is_synthetic=False)
        fold = FoldGraphs(1, 1, [graph], [graph], {})
        members = [Constant([2., 0.]), Constant([0., 1.]), Constant([0., 1.])]
        seeds = []
        with patch.object(runner, 'build_model', side_effect=members), \
             patch.object(runner, 'seed_fold', side_effect=seeds.append), \
             patch.object(runner, 'build_optimizer'), patch.object(runner, 'build_criterion'), \
             patch.object(runner, 'train_epoch') as trained, \
             patch.object(runner, 'evaluate', return_value=Evaluation(1.0, [0], [0])) as evaluation:
            result = runner.run_fold(fold, torch.device('cpu'),
                                     SynTrainingConfig(num_epochs=2, ensemble=3, center_logits=True, show_progress=False))
        self.assertEqual(seeds, [12345, 12346, 12347])
        self.assertEqual(trained.call_count, 6)
        self.assertEqual(result['optimizer_updates'], 6)
        ensemble = evaluation.call_args.args[0]
        self.assertIsInstance(ensemble, Ensemble)
        self.assertTrue(evaluation.call_args.kwargs['center_logits'])
        probabilities = ensemble(graph.x, graph.edge_index, torch.zeros(2, dtype=torch.long)).exp()
        expected = torch.stack([m.logits.softmax(0) for m in members]).mean(0)
        torch.testing.assert_close(probabilities[0], expected)
        with self.assertRaises(ValueError):
            SynTrainingConfig(ensemble=0).validate()


if __name__ == '__main__':
    unittest.main()
