"""Draw trial samples from already trained generators with other sampling settings.

Reads each selected job's checkpoint from a finished bundle and writes new
synthetic signals to `<output-root>/<job id>/subject_XX.npz`. The bundle is
not modified and nothing is retrained, so the samples are for inspection with
`evaluate.py --synthetic-root <output-root>`, not for classifier training.

Settings that can differ from the bundle's own generation:
    --cfg-scale   classifier-free guidance scale (the bundle used its saved one)
    --weights     'model' (best.pt, as the bundle used) or 'ema' (latest.pt)
    --steps       number of sampling steps, respaced from the trained schedule
    --trials-per-class, --seed
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'datasetsynthetic'

from .config import DEFAULT_OUTPUT
from .protocols import PROTOCOLS
from .storage import write_json


def resample_job(root, job, output_root, *, cfg_scale, weights, steps, trials_per_class, batch_size, seed, device):
    import torch
    from .eegdiffuser.diffusion import create_diffusion
    from .eegdiffuser.generate import sample_batch
    from .eegdiffuser.runtime import load_config, make_model, seed_all

    run = root / job['run_directory']
    subject = job['subject_id']
    saved = load_config(run)
    config = replace(saved, cfg_scale=cfg_scale, device=device)
    directory = run / 'checkpoints' / f'subject_{subject:02d}'
    model = make_model(config)
    if weights == 'model':
        checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
    else:
        # Locally produced resume checkpoint; it also holds optimizer and RNG state.
        checkpoint = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['ema'])
    epoch = checkpoint['epoch']
    del checkpoint
    model.eval()
    diffusion = create_diffusion(timestep_respacing=str(steps), noise_schedule=config.noise_schedule,
                                 diffusion_steps=config.diffusion_steps, learn_sigma=config.learn_sigma)
    seed_all(int(np.random.SeedSequence([seed, subject]).generate_state(1)[0]))
    labels = np.repeat(np.arange(config.num_classes, dtype=np.int64), trials_per_class)
    signals = []
    for start in range(0, len(labels), batch_size):
        batch = torch.tensor(labels[start:start + batch_size], device=device)
        signals.append(sample_batch(model, diffusion, batch, config))
        print(f'{job["job_id"]}: sampled {min(start + batch_size, len(labels))}/{len(labels)}', flush=True)
    target = Path(output_root) / job['job_id']
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / f'subject_{subject:02d}.npz', signals=np.concatenate(signals), labels=labels,
                        subject_ids=np.full(len(labels), subject, dtype=np.int64))
    return dict(job_id=job['job_id'], checkpoint_epoch=int(epoch), bundle_cfg_scale=saved.cfg_scale,
                bundle_steps=saved.diffusion_steps)


def resample(root, output_root, *, protocol='cross_subject', subjects=None, cfg_scale, weights='model',
             steps=250, trials_per_class=30, batch_size=10, seed=8888, device='cuda'):
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    if output_root == root or root in output_root.parents:
        raise ValueError('Write trial samples outside the bundle')
    if weights not in ('model', 'ema'):
        raise ValueError("weights must be 'model' or 'ema'")
    if cfg_scale < 0 or min(steps, trials_per_class, batch_size) < 1:
        raise ValueError('Invalid sampling settings')
    jobs = [job for job in json.loads((root / 'jobs.json').read_text()) if job['protocol'] == protocol
            and (not subjects or job['subject_id'] in subjects)]
    if not jobs:
        raise ValueError('No generation jobs selected')
    settings = dict(bundle=str(root), protocol=protocol, cfg_scale=cfg_scale, weights=weights, steps=steps,
                    trials_per_class=trials_per_class, seed=seed)
    done = [resample_job(root, job, output_root, cfg_scale=cfg_scale, weights=weights, steps=steps,
                         trials_per_class=trials_per_class, batch_size=batch_size, seed=seed, device=device)
            for job in jobs]
    write_json(output_root / 'resample_settings.json', dict(settings, jobs=done))
    return output_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output-root', type=Path, required=True, help='New folder outside the bundle')
    parser.add_argument('--protocol', choices=PROTOCOLS, default='cross_subject')
    parser.add_argument('--subjects', nargs='+', type=int, help='Generator subjects; default uses all')
    parser.add_argument('--cfg-scale', type=float, required=True)
    parser.add_argument('--weights', choices=('model', 'ema'), default='model')
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--trials-per-class', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=8888)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    try:
        target = resample(args.bundle, args.output_root, protocol=args.protocol, subjects=args.subjects,
                          cfg_scale=args.cfg_scale, weights=args.weights, steps=args.steps,
                          trials_per_class=args.trials_per_class, batch_size=args.batch_size, seed=args.seed,
                          device=args.device)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f'Trial samples saved under {target}')


if __name__ == '__main__':
    main()
