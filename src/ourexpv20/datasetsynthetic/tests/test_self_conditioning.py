"""Tiny self-conditioning integration checks; no real EEG or persistent outputs."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .test_verification import tiny_bundle, Config, generate, verify_generation, make_model, load_config
from .. import generate as cli
from ..eegdiffuser.diffusion import create_diffusion


class SelfConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = replace(Config(), subjects=(1,), device='cpu', epochs=1, batch_size=2,
            validation_per_class=1, samples_per_class=1, sample_batch_size=2,
            time_points=10, in_channels=2, embed_dim=8, num_heads=2, depth=1,
            mlp_ratio=1., diffusion_steps=21, num_global_tokens=2, self_condition=True)

    def test_cli_forwards_flag(self):
        with patch.object(cli, 'generate') as run:
            cli.main(['--protocol', 'within_subject', '--epochs', '1', '--self-conditioning'])
        self.assertIs(run.call_args.kwargs['self_conditioning'], True)

    def test_one_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'bundle'
            _, _, jobs = tiny_bundle(root)
            generate(root, protocols=('within_subject',), subjects=(1,), folds=(1,),
                     diffusion_config=self.config)
            self.assertTrue(verify_generation(root, jobs[0]))
            run = root / jobs[0]['run_directory']
            self.assertTrue(load_config(run).self_condition)
            with np.load(run / 'subject_01.npz') as data:
                self.assertEqual(data['signals'].shape, (2, 10, 2))
                self.assertTrue(np.isfinite(data['signals']).all())
            with self.assertRaisesRegex(ValueError, 'self-conditioning'):
                generate(root, protocols=('within_subject',), subjects=(1,), folds=(1,),
                         diffusion_config=replace(self.config, self_condition=False))

    def test_feedback_gradients_and_cfg_pairing(self):
        model = make_model(self.config)
        x, t, y = torch.randn(2, 2, 10), torch.tensor([1, 2]), torch.tensor([0, 1])
        feedback = torch.randn_like(x)
        with torch.no_grad():
            model.final_layer.linear.weight.normal_()
        model(x, t, y, x_self_cond=feedback).square().mean().backward()
        grad = model.x_self_cond_embedder.proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)
        model.eval()
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            model.forward_with_cfg(x, t, torch.tensor([0, 2]), 4., x_self_cond=feedback)
        paired = forward.call_args.kwargs['x_self_cond']
        self.assertTrue(torch.equal(paired[0], paired[1]))

    def test_ddim_reuses_previous_clean_prediction(self):
        model = make_model(self.config).eval()
        diffusion = create_diffusion('', diffusion_steps=21)
        feedback = []
        def record(module, args, kwargs):
            feedback.append(kwargs['x_self_cond'].detach().clone())
        handle = model.register_forward_pre_hook(record, with_kwargs=True)
        try:
            output = diffusion.ddim_sample_loop(model, (2, 2, 10),
                model_kwargs={'y': torch.tensor([0, 1])}, device=torch.device('cpu'))
        finally:
            handle.remove()
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(feedback), 21)
        self.assertEqual(feedback[0].abs().sum().item(), 0)
        self.assertGreater(feedback[1].abs().sum().item(), 0)
