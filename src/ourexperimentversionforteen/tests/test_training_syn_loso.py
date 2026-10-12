"""Ensure LOSO never uses held-out performance for epoch selection."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
from torch_geometric.data import Data

from src.ourexperimentversionforteen.datacustom import loso
from src.ourexperimentversionforteen.datacustom.dataset import FoldGraphs
from src.ourexperimentversionforteen.train.config_syn import SynTrainingConfig
from src.ourexperimentversionforteen.train import experiment_syn_loso as runner
from src.ourexperimentversionforteen.train.engine import Evaluation


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


class GeneratorCompatibilityTests(unittest.TestCase):
    def test_architecture_fields_allowed_but_wrong_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loader = loso.SavedLOSO.__new__(loso.SavedLOSO)
            loader.root = root
            loader.metadata = {'config': {'validation_per_class': 1}, 'artifact_sha256': {
                'inputs/metadata.json': 'meta', 'inputs/partitions.csv': 'parts',
                'original/subject_01.npz': 'signal'}}
            original = dict(trial_id='original:1', array_index='0', class_label='0',
                            subject_id='1', session_id='3', original_trial_index='2')
            other = dict(original, trial_id='original:2')
            loader.originals = {'original:1': original, 'original:2': other}
            job = dict(subject_id=1, run_directory='.', input_directory='inputs',
                       synthetic_trial_count=2, original_trial_ids=['original:1', 'original:2'],
                       training_trial_count=1, validation_trial_count=1)
            config = dict(subjects=[1], samples_per_class=1, validation_per_class=1,
                          epochs=100, model_architecture='external_architecture')
            meta = dict(status='complete', subjects={'1': {'epoch': 100}},
                        original_sha256={'metadata.json': 'meta', 'partitions.csv': 'parts',
                                         'subject_01.npz': 'signal'},
                        internal_partitions_sha256='internal', manifest_sha256='manifest',
                        generation={'1': {'sha256': 'generated', 'checkpoint_sha256': 'checkpoint'}})
            (root / 'config.json').write_text(json.dumps(config))
            (root / 'metadata.json').write_text(json.dumps(meta))
            generated = [dict(array_index=str(i), subject_id='1', class_label=str(i),
                              output_file='subject_01.npz', checkpoint_sha256='checkpoint',
                              source='synthetic', intended_partition='training') for i in range(2)]
            with patch.object(loso, 'verify_hash'), patch.object(loso, 'read_rows',
                    side_effect=lambda path: [dict(original, diffusion_subset='training'),
                                             dict(other, diffusion_subset='validation')]
                    if path.name == 'internal_partitions.csv' else generated):
                loader._verify_generator(job)
                meta['original_sha256']['subject_01.npz'] = 'wrong'
                (root / 'metadata.json').write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, 'different permitted originals'):
                    loader._verify_generator(job)


if __name__ == '__main__':
    unittest.main()
