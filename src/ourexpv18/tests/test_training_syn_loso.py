"""Ensure LOSO never uses held-out performance for epoch selection."""
import unittest
from unittest.mock import patch, MagicMock

import torch
from torch_geometric.data import Data

from src.ourexpv18.datacustom.dataset import FoldGraphs
from src.ourexpv18.train.config_syn import SynTrainingConfig
from src.ourexpv18.train import experiment_syn_loso as runner
from src.ourexpv18.train.engine import Evaluation


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


if __name__ == '__main__':
    unittest.main()
