"""Validation isolation, stopping, grouped splits and final-refit integration."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ..datascripts import build_subject_graphs, read_samples, dataset_config
from ..model import EEGGraphModel, ModelConfig
from ..train import TrainConfig, run_loso, run_within, within_config
from ..train.validation import EarlyStopping, Member, select_epochs, split_subjects
from ..train.within import fold_indices, nested_indices, partition_graphs
from ..train.loso import run_fold
from ..train.output import write_result, result_path
from ..train.run_loso import main as loso_main
from ..train.run_within import main as within_main
from .fixture import CONFIG, fake_recordings, write_bci2a_recordings
from .test_jia2019 import write_subject


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = TrainConfig(epochs=3, patience=1, ensemble=2, batch_size=8,
                                calibration_warmup_epochs=1, device='cpu', show_progress=False)
        with fake_recordings():
            cls.graphs = {s: build_subject_graphs('unused', s, CONFIG) for s in CONFIG.subjects}
            cls.samples = read_samples('unused', 1, CONFIG)
        cls.model_config = ModelConfig.for_data(CONFIG, paths=('temporal',))

    def test_defaults_and_invalid_settings(self):
        self.assertEqual((TrainConfig().epochs, TrainConfig().patience, within_config().epochs), (100, 15, 100))
        for bad in [dict(patience=0), dict(epochs=0), dict(min_delta=-1), dict(validation_fraction=1)]:
            with self.assertRaises(ValueError):
                replace(self.config, **bad).validate()
        # Validation may select an epoch before calibration begins; final refit must allow this.
        TrainConfig(epochs=1, calibration_warmup_epochs=10).validate()

    def test_subject_splits_all_datasets(self):
        for dataset, counts in [('liu2024',(36,10)), ('bci2a',(6,2)), ('jia2019',(11,3))]:
            subjects = dataset_config(dataset).subjects
            for held in subjects:
                train, val = split_subjects(subjects, held, 12345)
                self.assertEqual((len(train), len(val)), counts)
                self.assertFalse(set(train) & set(val))
                self.assertNotIn(held, train + val)
                self.assertEqual(set(train + val + [held]), set(subjects))
                self.assertEqual((train,val), split_subjects(subjects, held, 12345))

    def test_patience_ties_and_improvement(self):
        stop = EarlyStopping(2)
        self.assertFalse(stop.update(1,.6))
        self.assertFalse(stop.update(2,.6))
        self.assertFalse(stop.update(3,.7))
        self.assertFalse(stop.update(4,.7))
        self.assertTrue(stop.update(5,.65))
        self.assertEqual((stop.best_epoch, stop.best_score), (3,.7))
        with self.assertRaises(ValueError): stop.update(6,float('nan'))

    def test_within_outer_folds_preserved_and_trials_disjoint(self):
        trials = np.arange(40); labels = [0,1] * 20
        outer = fold_indices(trials, labels, 10, 42)
        nested = list(nested_indices(trials, labels, 10, 42))
        tested=[]
        for k,(train,val,test,non_test) in enumerate(nested):
            np.testing.assert_array_equal(test,outer[k][1])
            np.testing.assert_array_equal(val,outer[(k+1)%10][1])
            self.assertEqual(set(train) | set(val),set(non_test))
            self.assertFalse(set(trials[train]) & set(trials[val]))
            self.assertFalse(set(trials[train]) & set(trials[test]))
            self.assertFalse(set(trials[val]) & set(trials[test]))
            tested.extend(test)
        self.assertEqual(sorted(tested),list(range(40)))
        with self.assertRaises(ValueError): list(nested_indices(trials,labels,2))

    def test_alignment_does_not_fit_validation_or_test_signal(self):
        original = partition_graphs(self.samples, [0,1,2,3], [0,1,4,5], CONFIG)
        changed = list(self.samples)
        for i in [4,5,6,7]:
            changed[i] = replace(changed[i], signal=changed[i].signal * 100)
        other = partition_graphs(changed, [0,1,2,3], [0,1,4,5], CONFIG)
        for i in [0,1]:
            torch.testing.assert_close(original[i].signal, other[i].signal)
            torch.testing.assert_close(original[i].edge_attr, other[i].edge_attr)

    def test_calibration_never_reads_target_labels(self):
        target = self.graphs[2]
        flipped = [g.clone() for g in target]
        for g in flipped: g.y = 1 - g.y
        a=Member(self.graphs[1],target,self.model_config,self.config,123,torch.device('cpu'))
        b=Member(self.graphs[1],flipped,self.model_config,self.config,123,torch.device('cpu'))
        for epoch in range(1,3):
            torch.testing.assert_close(a.epoch(epoch),b.epoch(epoch))
        self.assertTrue(all('y' not in g for g in a.calibration.graphs))

    def test_selection_stops_and_each_target_has_own_ensemble(self):
        config=replace(self.config,epochs=10,patience=2)
        with patch('src.GTPGTA.train.validation.classification_metrics', return_value={'balanced_accuracy':.5}):
            result=select_epochs(self.graphs[1], {2:self.graphs[2],3:self.graphs[3]},self.model_config,
                                 config,torch.device('cpu'))
        self.assertEqual((result['selected_epoch'],result['stopped_epoch']), (1,3))
        self.assertTrue(result['independent_target_ensembles'])
        self.assertEqual(set(result['history'][0]['by_group']),{'2','3'})

    def test_final_refit_uses_selected_count_and_test_labels_only_at_endpoint(self):
        target=[g.clone() for g in self.graphs[2]]
        for g in target: g.y=1-g.y
        cfg=replace(self.config,epochs=2)
        a=run_fold(self.graphs[1],self.graphs[2],'test',CONFIG,self.model_config,cfg,torch.device('cpu'))
        b=run_fold(self.graphs[1],target,'test',CONFIG,self.model_config,cfg,torch.device('cpu'))
        self.assertEqual(a['scores'],b['scores'])
        self.assertEqual(len(a['history'][0]['train_loss']),2)
        self.assertNotIn('highest_epoch',a)
        self.assertNotIn('test_accuracy_per_epoch',a)

    def test_loso_saved_splits_refit_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp, fake_recordings():
            models=Path(temp)/'models'
            result=run_loso('unused',CONFIG,self.config,held_out_subjects=(2,),models_dir=models,paths=('temporal',))
            f=result['subjects']['2']; split=f['validation']['split']
            self.assertNotIn(2,split['training_subjects']+split['validation_subjects'])
            self.assertEqual(f['refit_subjects'],[1,3])
            self.assertEqual(f['training_samples'],16)
            self.assertEqual(f['selected_epoch'],f['validation']['selected_epoch'])
            checkpoint=torch.load(models/'S2_model1.pt',weights_only=True)
            self.assertEqual(checkpoint['train_config']['epochs'],f['selected_epoch'])
            self.assertEqual(checkpoint['validation']['split'],split)
            model=EEGGraphModel(ModelConfig(**checkpoint['model_config']))
            model.load_state_dict(checkpoint['state_dict'])
            out=result_path(temp,'liu2024','LOSO',12345)
            write_result(result,out)
            self.assertTrue(out.name.startswith('Version29_liu2024_LeftRight8s_'))
            self.assertNotIn('highest_epoch',json.loads(out.read_text())['summary'])

    def test_within_run_refits_on_train_plus_validation(self):
        cfg=within_config(epochs=2,patience=1,ensemble=1,batch_size=8,device='cpu',show_progress=False)
        with fake_recordings():
            result=run_within('unused',CONFIG,cfg,subjects=(1,),folds=4,paths=('temporal',))
        folds=result['subjects']['1']['folds']
        self.assertEqual(sorted(i for f in folds for i in f['test_index']),list(range(8)))
        for f in folds:
            self.assertEqual(f['training_samples'],6)
            self.assertEqual(f['validation']['training_samples'],4)
            self.assertEqual(f['selected_epoch'],len(f['history'][0]['train_loss']))
            self.assertNotIn('calibration',f)

    def test_bci_clis_both_protocols(self):
        with tempfile.TemporaryDirectory() as temp:
            write_bci2a_recordings(temp,subjects=range(1,10))
            common=['--dataset','bci2a','--data-dir',temp,'--output-dir',temp,'--subjects','1',
                    '--epochs','1','--patience','1','--device','cpu','--quiet','--paths','temporal','--save-models']
            loso_main(common+['--no-target-calibration'])
            within_main(common+['--folds','3'])
            files=list((Path(temp)/'bci2a').glob('*.json'))
            self.assertEqual(len(files),2)
            for path in files:
                result=json.loads(path.read_text())
                self.assertEqual(result['train_config']['epochs'],1)
                self.assertTrue(result['subjects']['1'])

    def test_jia_within_supported_but_marked_confounded(self):
        with tempfile.TemporaryDirectory() as temp:
            write_subject(temp,1,trials=3,channels=5)
            dc=replace(dataset_config('jia2019'),subjects=(1,),num_channels=5)
            cfg=within_config(epochs=1,ensemble=1,batch_size=4,device='cpu',show_progress=False)
            with self.assertWarnsRegex(UserWarning,'confounded'):
                result=run_within(temp,dc,cfg,subjects=(1,),folds=3,paths=('temporal',))
            self.assertFalse(result['validity']['valid_for_mi_comparison'])
            self.assertEqual(len(result['subjects']['1']['folds']),3)


if __name__ == '__main__':
    unittest.main()
