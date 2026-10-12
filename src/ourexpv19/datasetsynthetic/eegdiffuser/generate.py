"""Generate labeled EEG from each subject's best ordinary-model checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'eegdiffuser'

from .config import Config
from .data import read_rows, write_rows
from .runtime import load_config, make_diffusion, make_model, now, seed_all, sha256, write_json


@torch.no_grad()
def sample_batch(model, diffusion, labels: torch.Tensor, config: Config) -> np.ndarray:
    noise = torch.randn(len(labels), config.in_channels, config.time_points, device=config.device)
    noise = torch.cat([noise, noise], dim=0)
    null_labels = torch.full_like(labels, config.num_classes)
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, noise.shape, noise, clip_denoised=False,
        model_kwargs=dict(y=torch.cat([labels, null_labels]), cfg_scale=config.cfg_scale),
        progress=False, device=torch.device(config.device))
    samples, _ = samples.chunk(2, dim=0)
    result = (samples * config.signal_scale).transpose(1, 2).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Generated signals contain non-finite values')
    return result


def generate_subject(run: Path, subject: int, config: Config) -> None:
    metadata = json.loads((run / 'metadata.json').read_text())
    if config != load_config(run) or subject not in config.subjects:
        raise ValueError('Generation must use the saved run configuration and subjects')
    best = run / 'checkpoints' / f'subject_{subject:02d}' / 'best.pt'
    digest = sha256(best)
    if sha256(run / 'internal_partitions.csv') != metadata['internal_partitions_sha256']:
        raise ValueError('Saved internal partitions changed')
    output = run / f'subject_{subject:02d}.npz'
    existing = metadata.get('generation', {}).get(str(subject))
    if existing:
        if sha256(output) != existing['sha256'] or digest != existing['checkpoint_sha256']:
            raise ValueError('Existing generation or source checkpoint changed; use a new run')
        if sha256(run / 'manifest.csv') != metadata['manifest_sha256']:
            raise ValueError('Synthetic manifest changed')
        print(f'Subject {subject}: reused verified synthetic data', flush=True)
        return
    if metadata['subjects'].get(str(subject), {}).get('epoch') != config.epochs:
        raise ValueError('Complete the configured training epochs before generation')
    checkpoint = torch.load(best, map_location='cpu', weights_only=True)
    if checkpoint['config'] != config.to_dict() or checkpoint['subject_id'] != subject or checkpoint['partitions_sha256'] != metadata['internal_partitions_sha256']:
        raise ValueError('Best checkpoint does not match this run')
    model = make_model(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    diffusion = make_diffusion(config)
    generation_seed = int(np.random.SeedSequence([config.generation_seed, subject]).generate_state(1)[0])
    seed_all(generation_seed)
    labels = np.repeat(np.arange(config.num_classes, dtype=np.int64), config.samples_per_class)
    signals = []
    for start in range(0, len(labels), config.sample_batch_size):
        batch_labels = torch.tensor(labels[start:start + config.sample_batch_size], device=config.device)
        signals.append(sample_batch(model, diffusion, batch_labels, config))
        print(f'Subject {subject}: generated {min(start + config.sample_batch_size, len(labels))}/{len(labels)}', flush=True)
    array = np.concatenate(signals)
    temporary = output.with_name(output.name + '.tmp')
    trial_ids = np.array([f'S{subject:02d}_synthetic_{i:05d}' for i in range(len(labels))])
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, signals=array, labels=labels,
                            subject_ids=np.full(len(labels), subject, dtype=np.int64), synthetic_trial_ids=trial_ids)
    temporary.replace(output)
    manifest = run / 'manifest.csv'
    rows = [r for r in read_rows(manifest) if int(r['subject_id']) != subject] if manifest.exists() else []
    rows.extend(dict(subject_id=str(subject), synthetic_trial_id=str(trial_ids[i]), class_label=str(int(label)),
                     array_index=str(i), output_file=output.name, generation_seed=str(generation_seed),
                     checkpoint=str(best.relative_to(run)), checkpoint_sha256=digest,
                     checkpoint_epoch=str(checkpoint['epoch']), source='synthetic', intended_partition='training')
                for i, label in enumerate(labels))
    write_rows(manifest, rows)
    metadata.setdefault('generation', {})[str(subject)] = dict(sha256=sha256(output), checkpoint_sha256=digest,
        checkpoint_epoch=checkpoint['epoch'], generation_seed=generation_seed, trial_count=len(labels),
        counts={str(label): config.samples_per_class for label in range(config.num_classes)},
        shape=list(array.shape), minimum=float(array.min()), maximum=float(array.max()),
        mean=float(array.mean()), std=float(array.std()), finished_utc=now())
    metadata['manifest_sha256'] = sha256(manifest)
    metadata['subjects'][str(subject)]['status'] = 'complete'
    write_json(run / 'metadata.json', metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    config = load_config(run)
    if config.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(config.device).index or 0)
    for subject in config.subjects:
        generate_subject(run, subject, config)
    metadata = json.loads((run / 'metadata.json').read_text())
    metadata.update(status='complete', finished_utc=now())
    write_json(run / 'metadata.json', metadata)


if __name__ == '__main__':
    main()
