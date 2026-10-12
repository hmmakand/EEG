"""Ensure LOSO never uses held-out performance for epoch selection."""
import unittest
from unittest.mock import patch, MagicMock

import torch
from torch_geometric.data import Data

from src.ourexpv22.datacustom.dataset import FoldGraphs
from src.ourexpv22.train.config_syn import SynTrainingConfig
from src.ourexpv22.train import experiment_syn_loso as runner
from src.ourexpv22.train.engine import Evaluation


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
             patch.object(runner, 'train_epoch', side_effect=lambda *a, **k: events.append('train')), \
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
        from src.ourexpv22.train.setup import Ensemble

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

    def test_subject_invariance_losses(self):
        from src.ourexpv22.train.invariance import SubjectInvariance
        torch.manual_seed(0)
        features = torch.randn(12, 5, requires_grad=True)
        labels = torch.tensor([0, 1] * 6)
        subjects = torch.tensor([2, 2, 2, 2, 5, 5, 5, 5, 9, 9, 9, 9])
        adversarial = SubjectInvariance('adversarial', 5, {2, 5, 9}, weight=0.5)
        loss = adversarial(features, labels, subjects, progress=1.0)
        loss.backward()
        head_gradient = [p.grad.clone() for p in adversarial.head.parameters()]
        reversed_gradient = features.grad.clone()
        # The same loss without reversal gives the opposite feature gradient,
        # scaled by the ramped weight; the subject classifier itself is unaffected.
        features.grad = None
        plain = torch.nn.functional.cross_entropy(adversarial.head(adversarial.norm(features)), adversarial.lookup[subjects])
        plain.backward()
        strength = 0.5 * (2 / (1 + torch.exp(torch.tensor(-10.))) - 1)
        torch.testing.assert_close(reversed_gradient, -strength * features.grad, rtol=1e-4, atol=1e-6)
        self.assertGreater(sum(float(g.abs().sum()) for g in head_gradient), 0)
        features.grad = None
        adversarial(features, labels, subjects, progress=0.0).backward()
        self.assertEqual(float(features.grad.abs().sum()), 0.0)

        contrastive = SubjectInvariance('contrastive', 5, {2, 5, 9}, weight=1.0)
        self.assertGreater(float(contrastive(features, labels, subjects, 0.5)), 0)
        # One subject per batch offers no cross-subject positives.
        self.assertEqual(float(contrastive(features[:4], labels[:4], subjects[:4], 0.5)), 0.0)
        with self.assertRaises(ValueError):
            adversarial(features, labels, torch.full((12,), 7), 1.0)
        for bad in (dict(invariance='other'), dict(invariance_weight=-1.0)):
            with self.assertRaises(ValueError):
                SynTrainingConfig(**bad).validate()

    def test_proxy_loss_and_test_time_adaptation(self):
        from src.ourexpv22.train.invariance import SubjectInvariance
        from src.ourexpv22.train import adaptation
        torch.manual_seed(0)
        proxy = SubjectInvariance('proxy', 5, {2, 5}, weight=1.0)
        features = torch.randn(8, 5, requires_grad=True)
        loss = proxy(features, torch.tensor([0, 1] * 4), torch.tensor([2] * 4 + [5] * 4), 0.5)
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertGreater(float(proxy.proxies.grad.abs().sum()), 0)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm, self.out = torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2)

            def forward(self, x, edge_index, batch, return_features=False):
                features = self.norm(x)
                return (self.out(features), features) if return_features else self.out(features)

        from torch_geometric.loader import DataLoader
        graphs = [Data(x=torch.randn(1, 2) + 5, edge_index=torch.empty((2, 0), dtype=torch.long), y=torch.tensor(-7))
                  for _ in range(24)]  # labels are deliberately invalid: adaptation must not read them
        loader = DataLoader(graphs, batch_size=8)
        model = Tiny().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for method in adaptation.METHODS:
            adapted = adaptation.adapted_copy(method, model, loader, torch.device('cpu'))
            batch = next(iter(loader))
            self.assertEqual(tuple(adaptation.predict(adapted, batch).shape), (8, 2))
            self.assertFalse(adapted.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        shifted = adaptation.adapted_copy('batchnorm', model, loader, torch.device('cpu'))
        self.assertGreater(float(shifted.norm.running_mean.mean()), 4.0)
        with self.assertRaises(ValueError):
            adaptation.adapt('other', model, loader, torch.device('cpu'))

    def test_invariance_is_built_and_passed_to_training(self):
        graphs = [Data(x=torch.ones(2, 8), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([i % 2]),
                       sample_id=str(i), is_synthetic=False, subject_id=subject)
                  for i, subject in enumerate((2, 2, 3, 3))]
        fold = FoldGraphs(1, 1, graphs, graphs[:1], {})
        model = MagicMock(feature_dim=6)
        with patch.object(runner, 'build_model', return_value=model), \
             patch.object(runner, 'build_optimizer') as optimizer, patch.object(runner, 'build_criterion'), \
             patch.object(runner, 'train_epoch') as trained, \
             patch.object(runner, 'evaluate', return_value=Evaluation(1.0, [0], [0])):
            runner.run_fold(fold, torch.device('cpu'), SynTrainingConfig(num_epochs=2, invariance='adversarial',
                                                                         invariance_weight=0.3, show_progress=False))
        invariance = trained.call_args.kwargs['invariance']
        self.assertEqual((invariance.mode, invariance.weight, invariance.head[-1].out_features), ('adversarial', 0.3, 2))
        self.assertEqual([call.kwargs['progress'] for call in trained.call_args_list], [0.0, 0.5])
        self.assertEqual(len(list(optimizer.call_args.args[2])), len(list(invariance.parameters())))

    def test_use_synthetic_false_trains_on_originals_only(self):
        def graph(sample_id, synthetic):
            return Data(x=torch.ones(2, 8), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([0]),
                        sample_id=sample_id, is_synthetic=synthetic)
        fold = FoldGraphs(1, 1, [graph('a', False), graph('b', True), graph('c', False)], [graph('t', False)], {})
        seen = []
        with patch.object(runner, 'build_model', return_value=MagicMock()), \
             patch.object(runner, 'build_optimizer'), patch.object(runner, 'build_criterion'), \
             patch.object(runner, 'train_epoch', side_effect=lambda model, loader, *rest, **k: seen.append(len(loader.dataset))), \
             patch.object(runner, 'evaluate', return_value=Evaluation(1.0, [0], [0])):
            kept = runner.run_fold(fold, torch.device('cpu'), SynTrainingConfig(num_epochs=1, show_progress=False))
            dropped = runner.run_fold(fold, torch.device('cpu'),
                                      SynTrainingConfig(num_epochs=1, use_synthetic=False, show_progress=False))
        self.assertEqual(seen, [3, 2])
        self.assertEqual(kept['counts']['training']['synthetic'], 1)
        self.assertEqual(dropped['counts']['training']['synthetic'], 0)
        self.assertEqual(dropped['sample_ids']['training'], ['a', 'c'])


if __name__ == '__main__':
    unittest.main()
