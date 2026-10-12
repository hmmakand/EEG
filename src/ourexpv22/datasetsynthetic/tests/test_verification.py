"""Tiny CPU verification: saved folds, architecture checks, and one generation job."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasetsynthetic.config import PreparationConfig
from datasetsynthetic.protocols import save_protocols, validate_plans, PROTOCOLS
from datasetsynthetic.storage import read_rows, write_rows, write_json, sha256
from datasetsynthetic.generate import generate, verify_generation, finalize
from datasetsynthetic.eegdiffuser.config import Config
from datasetsynthetic.eegdiffuser.runtime import load_config, make_model
from datasetsynthetic.eegdiffuser import train as training


def tiny_bundle(root):
    """Two subjects, eight short trials each; real fold preparation, no MATLAB needed."""
    prep = PreparationConfig(subjects=(1, 2), n_folds=2, samples_per_class=1, validation_per_class=1)
    prep = replace(prep, trials=replace(prep.trials, channel_indices=(0, 1), duration_seconds=.04))
    original = root / 'original'
    original.mkdir(parents=True)
    rows = []
    # Each half chosen by KFold(seed=42) has two samples of each class.
    labels = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int64)
    rng = np.random.default_rng(42)
    for subject in prep.subjects:
        name = f'subject_{subject:02d}.npz'
        np.savez_compressed(original / name, signals=rng.normal(size=(8, 10, 2)), labels=labels,
                            subject_ids=np.full(8, subject))
        rows.extend(dict(subject_id=subject, source_file=f'A{subject:02d}T.mat', session_id=3,
                         original_trial_index=i, array_index=i, class_label=int(labels[i]),
                         trial_id=f'original:S{subject:02d}:3:{i}', signal_file=f'original/{name}') for i in range(8))
    write_rows(original / 'trials.csv', rows)
    jobs = save_protocols(root, rows, prep)
    write_json(root / 'metadata.json', dict(format_version=2, workflow='fold_first_signals', config=asdict(prep),
        status='prepared', synthetic_generation_complete=False,
        artifact_sha256={str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()}))
    return prep, rows, jobs


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / 'bundle'
        cls.prep, cls.originals, cls.jobs = tiny_bundle(cls.root)
        cls.config = replace(Config(), subjects=(1,), device='cpu', epochs=1, batch_size=2,
                             validation_per_class=1, samples_per_class=1, sample_batch_size=2,
                             time_points=10, in_channels=2, embed_dim=8, num_heads=2,
                             depth=1, mlp_ratio=1., diffusion_steps=21, num_global_tokens=2)
        # One model, one optimizer update, one validation batch, two sampled signals.
        cls.job = cls.jobs[0]
        cls.run_dir = cls.root / cls.job['run_directory']
        with redirect_stdout(io.StringIO()), patch.object(training, 'train_epoch', wraps=training.train_epoch) as epoch:
            cls.status = generate(cls.root, protocols=('within_subject',), subjects=(1,), folds=(1,),
                                  diffusion_config=cls.config)
            cls.epoch_calls = epoch.call_count

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_one_job_real_training_sampling_and_saved_config(self):
        self.assertEqual(self.epoch_calls, 1)
        self.assertTrue(verify_generation(self.root, self.job))
        config = load_config(self.run_dir)
        self.assertEqual(config.num_global_tokens, 2)
        self.assertEqual(config.model_kwargs()['num_global_tokens'], 2)
        self.assertEqual(tuple(make_model(config).global_queries.shape), (1, 2, 8))
        with np.load(self.run_dir / 'subject_01.npz') as a:
            self.assertEqual(a['signals'].shape, (2, 10, 2))
            self.assertTrue(np.isfinite(a['signals']).all())
            np.testing.assert_array_equal(a['labels'], [0, 1])
        self.assertEqual(self.status['within_subject']['jobs_complete'], 1)
        self.assertEqual(self.status['cross_subject']['jobs_complete'], 0)
        self.assertFalse((self.root / 'protocols/within_subject.csv').exists())
        self.assertFalse((self.root / 'protocols/cross_subject.csv').exists())
        internal = read_rows(self.run_dir / 'internal_partitions.csv')
        self.assertEqual(sum(r['diffusion_subset'] == 'training' for r in internal), 2)
        self.assertEqual(sum(r['diffusion_subset'] == 'validation' for r in internal), 2)

    def test_new_global_fusion_model_forward_gradients_and_guidance(self):
        torch.manual_seed(42)
        model = make_model(self.config)
        self.assertEqual(len(model.global_fusions), 1)
        x = torch.randn(1, 2, 10)
        t, y = torch.tensor([3]), torch.tensor([0])
        target = torch.randn(1, 4, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        # The zero output layer and zero fusion gate deliberately delay gradients.
        # Three tiny updates let the output, gate, and fusion branch all activate.
        for _ in range(3):
            optimizer.zero_grad()
            output = model(x, t, y)
            self.assertEqual(tuple(output.shape), (1, 4, 10))
            loss = (output - target).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            optimizer.step()
        for parameter in (model.global_queries, model.global_fusions[0].write_gate,
                          model.global_fusions[0].global_read.in_proj_weight,
                          model.global_fusions[0].patch_write.in_proj_weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)
        model.eval()
        with torch.no_grad():
            guided = model.forward_with_cfg(x.repeat(2, 1, 1), t.repeat(2), torch.tensor([0, 2]), 4.)
        self.assertEqual(tuple(guided.shape), (2, 4, 10))
        self.assertTrue(torch.isfinite(guided).all())

    def test_both_protocols_isolate_generators_from_testing(self):
        plans = {p: read_rows(self.root / 'protocols' / f'{p}.plan.csv') for p in PROTOCOLS}
        validate_plans(plans, self.jobs, self.originals, self.prep)
        for protocol, rows in plans.items():
            for target, fold in {(r['target_subject'], r['fold']) for r in rows}:
                selected = [r for r in rows if (r['target_subject'], r['fold']) == (target, fold)]
                test = {r['trial_id'] for r in selected if r['role'] == 'testing'}
                for r in selected:
                    if r['source'] == 'synthetic':
                        job = next(j for j in self.jobs if j['job_id'] == r['generator_job'])
                        self.assertFalse(set(job['original_trial_ids']) & test)
                        if protocol == 'cross_subject':
                            self.assertNotEqual(int(target), job['subject_id'])

    def test_completed_job_reuse_does_not_train(self):
        with patch.object(training, 'run_training', side_effect=AssertionError('must reuse')), redirect_stdout(io.StringIO()):
            generate(self.root, protocols=('within_subject',), subjects=(1,), folds=(1,), diffusion_config=self.config)

    def test_old_config_rejected_by_verify_finalize_and_resume(self):
        path = self.run_dir / 'config.json'
        before = path.read_bytes()
        try:
            old = json.loads(before)
            del old['model_architecture']
            del old['num_global_tokens']
            write_json(path, old)
            for action in (lambda: verify_generation(self.root, self.job), lambda: finalize(self.root),
                           lambda: training.prepare_run(self.root / self.job['input_directory'], self.run_dir, self.config, resume=True)):
                with self.assertRaisesRegex(ValueError, 'architecture'):
                    action()
            with patch.object(training, 'run_training', side_effect=AssertionError('must reject before training')):
                # Even requesting fresh cross-subject jobs preflights incompatible completed within-subject jobs.
                with self.assertRaisesRegex(ValueError, 'architecture'):
                    generate(self.root, protocols=('cross_subject',), diffusion_config=self.config)
        finally:
            path.write_bytes(before)

    def test_model_source_change_rejected(self):
        path = self.run_dir / 'metadata.json'
        before = path.read_bytes()
        try:
            metadata = json.loads(before)
            metadata['source_sha256']['models/eegdiffuser.py'] = 'old_model'
            write_json(path, metadata)
            with self.assertRaisesRegex(ValueError, 'Model source changed'):
                verify_generation(self.root, self.job)
        finally:
            path.write_bytes(before)

    def test_tokens_must_be_saved_valid_and_match_existing_run(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'num_global_tokens'):
                replace(self.config, num_global_tokens=value).validate()
        with self.assertRaisesRegex(ValueError, 'global-token count'):
            generate(self.root, protocols=('within_subject',), subjects=(1,), folds=(1,),
                     diffusion_config=replace(self.config, num_global_tokens=3))


if __name__ == '__main__':
    unittest.main()
