"""Train local EEGDiffuser models on saved original training partitions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'eegdiffuser'

from .config import Config, ORIGINAL, SYNTHETIC
from .data import original_fingerprints, prepare_internal_partitions, subject_loaders
from .runtime import (load_config, make_diffusion, make_model, now, restore_rng,
                      rng_state, save_checkpoint, seed_all, sha256, write_json)


@torch.no_grad()
def update_ema(ema, model, decay: float = 0.9999) -> None:
    parameters = dict(model.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.mul_(decay).add_(parameters[name].data, alpha=1 - decay)


def train_epoch(model, loader, optimizer, scheduler, ema, diffusion, config: Config) -> float:
    # Documented reference correction: validation must not disable future label dropout.
    model.train()
    losses = []
    for x, y in loader:
        optimizer.zero_grad()
        x, y = x.to(config.device), y.to(config.device)
        timesteps = torch.randint(0, diffusion.num_timesteps, (x.shape[0],)).to(config.device)
        loss = diffusion.training_losses(model, x, timesteps, dict(y=y))['loss'].mean()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(ema, model, config.ema_decay)
        scheduler.step()
        losses.append(loss.detach().cpu().numpy())
    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, diffusion, config: Config) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(config.device), y.to(config.device)
        generator = torch.Generator().manual_seed(i)
        timesteps = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), generator=generator).to(config.device)
        # Reference behavior: timestep seed is fixed per batch; noise is freshly sampled.
        loss = diffusion.training_losses(model, x, timesteps, dict(y=y))['loss'].mean()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite validation loss')
        losses.append(loss.detach().cpu().numpy())
    return float(np.mean(losses))


def prepare_run(original: Path, run: Path, config: Config, resume: bool = False) -> dict:
    config.validate()
    if config.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; explicitly choose --device cpu')
        torch.cuda.set_device(torch.device(config.device).index or 0)
    fingerprints = original_fingerprints(original, config)
    if (run / 'config.json').exists():
        if not resume:
            raise ValueError('Run already exists; use --resume or a new --run-dir')
        if load_config(run) != config:
            raise ValueError('Resume settings must exactly match saved config.json')
        metadata = json.loads((run / 'metadata.json').read_text())
        if metadata['original_sha256'] != fingerprints:
            raise ValueError('Original data changed since this run was created')
        if sha256(run / 'internal_partitions.csv') != metadata['internal_partitions_sha256']:
            raise ValueError('Internal partitions changed since this run was created')
    else:
        if resume:
            raise ValueError('Cannot resume a run without config.json')
        if run.exists() and any(run.iterdir()):
            raise ValueError('New run directory must be empty')
        run.mkdir(parents=True, exist_ok=True)
        prepare_internal_partitions(original, run, config)
        root = Path(__file__).resolve().parent
        metadata = dict(created_utc=now(), original_directory=str(original),
                        original_sha256=fingerprints, internal_partitions_sha256=sha256(run / 'internal_partitions.csv'),
                        source_sha256={str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*.py'))},
                        packages={name: version(name) for name in ('torch', 'numpy', 'timm')},
                        python=platform.python_version(), device=config.device,
                        device_name=torch.cuda.get_device_name() if config.device.startswith('cuda') else 'CPU',
                        original_partition_used='training', original_validation_and_testing_used=False,
                        internal_split='5 trials per class by default; PCG64 SeedSequence([42, subject, stage, class/subset])',
                        model_selection='lowest internal validation loss; ordinary model, not EMA',
                        reference_correction='restore model.train() at the start of every epoch',
                        status='prepared', subjects={})
        write_json(run / 'config.json', config.to_dict())
        write_json(run / 'metadata.json', metadata)
    prepare_internal_partitions(original, run, config)
    return metadata


def train_subject(original: Path, run: Path, subject: int, config: Config,
                  stop_after_epoch: int | None = None) -> bool:
    """Resume at an epoch boundary; return whether all configured epochs completed."""
    seed_all(config.seed)
    train_loader, val_loader = subject_loaders(original, run, subject, config)
    model = make_model(config)
    ema = deepcopy(model)
    ema.requires_grad_(False)
    update_ema(ema, model, decay=0)
    ema.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * len(train_loader), eta_min=config.min_learning_rate)
    diffusion = make_diffusion(config)
    checkpoint_dir = run / 'checkpoints' / f'subject_{subject:02d}'
    latest_path, best_path = checkpoint_dir / 'latest.pt', checkpoint_dir / 'best.pt'
    start_epoch, best_epoch, best_loss, history = 0, 0, math.inf, []
    if latest_path.exists():
        # Only load locally produced, trusted resume checkpoints (includes RNG tuples).
        state = torch.load(latest_path, map_location='cpu', weights_only=False)
        if state['config'] != config.to_dict() or state['subject_id'] != subject or state['partitions_sha256'] != sha256(run / 'internal_partitions.csv'):
            raise ValueError('Checkpoint configuration, subject, or partitions do not match')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best_epoch, best_loss, history = state['epoch'], state['best_epoch'], state['best_loss'], state['history']
        restore_rng(state['rng'])
        # A crash can leave best.pt ahead of the last durable training epoch.
        # Restore the matching best snapshot before replaying unsaved epochs.
        if not best_path.exists() or sha256(best_path) != state['best_sha256']:
            save_checkpoint(best_path, state['best_checkpoint'])
        del state
    target_epoch = min(config.epochs, stop_after_epoch) if stop_after_epoch is not None else config.epochs
    print(f'Subject {subject}: {len(train_loader.dataset)} training, {len(val_loader.dataset)} internal validation; epochs {start_epoch + 1}..{target_epoch}', flush=True)
    for epoch in range(start_epoch + 1, target_epoch + 1):
        started = time.monotonic()
        training_loss = train_epoch(model, train_loader, optimizer, scheduler, ema, diffusion, config)
        validation_loss = validate(model, val_loader, diffusion, config)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            save_checkpoint(best_path, dict(model=model.state_dict(), epoch=epoch, validation_loss=best_loss,
                                           config=config.to_dict(), subject_id=subject,
                                           partitions_sha256=sha256(run / 'internal_partitions.csv')))
        history.append(dict(epoch=epoch, training_loss=training_loss, validation_loss=validation_loss,
                            learning_rate=optimizer.param_groups[0]['lr'], seconds=time.monotonic() - started))
        if epoch % config.checkpoint_interval == 0 or epoch == target_epoch:
            save_checkpoint(latest_path, dict(model=model.state_dict(), ema=ema.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch,
                best_epoch=best_epoch, best_loss=best_loss, history=history, rng=rng_state(),
                best_checkpoint=torch.load(best_path, map_location='cpu', weights_only=True),
                best_sha256=sha256(best_path),
                config=config.to_dict(), subject_id=subject,
                partitions_sha256=sha256(run / 'internal_partitions.csv')))
        write_json(run / 'history' / f'subject_{subject:02d}.json', history)
        metadata = json.loads((run / 'metadata.json').read_text())
        metadata['subjects'][str(subject)] = dict(epoch=epoch, best_epoch=best_epoch,
            best_validation_loss=best_loss, status='trained' if epoch == config.epochs else 'training')
        metadata['updated_utc'] = now()
        write_json(run / 'metadata.json', metadata)
        print(f'S{subject} epoch {epoch}/{config.epochs}: train={training_loss:.6f} val={validation_loss:.6f} best_epoch={best_epoch} seconds={history[-1]["seconds"]:.2f}', flush=True)
    return target_epoch >= config.epochs


def run_training(original: Path, run: Path, config: Config, *, resume=False,
                 stop_after_epoch: int | None = None, generate=False) -> None:
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError('stop-after-epoch must be positive')
    original, run = original.resolve(), run.resolve()
    metadata = prepare_run(original, run, config, resume)
    metadata['status'] = 'training'
    write_json(run / 'metadata.json', metadata)
    try:
        for subject in config.subjects:
            complete = train_subject(original, run, subject, config, stop_after_epoch)
            if not complete:
                metadata = json.loads((run / 'metadata.json').read_text())
                metadata['status'] = 'paused'
                write_json(run / 'metadata.json', metadata)
                return
            if generate:
                from .generate import generate_subject
                generate_subject(run, subject, config)
        metadata = json.loads((run / 'metadata.json').read_text())
        metadata['status'] = 'complete' if generate else 'trained'
        metadata['finished_utc'] = now()
        write_json(run / 'metadata.json', metadata)
    except Exception as error:
        metadata = json.loads((run / 'metadata.json').read_text())
        metadata.update(status='failed', error=str(error), updated_utc=now())
        write_json(run / 'metadata.json', metadata)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dir', type=Path, default=ORIGINAL)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device')
    parser.add_argument('--samples-per-class', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--generate', action='store_true', help='Generate after each subject finishes training')
    parser.add_argument('--stop-after-epoch', type=int, help='Pause after this epoch without changing scheduler horizon')
    args = parser.parse_args(argv)
    if args.resume and args.run_dir is None:
        parser.error('--resume requires --run-dir')
    run = args.run_dir or SYNTHETIC / datetime.now(timezone.utc).strftime('eegdiffuser_%Y%m%dT%H%M%SZ')
    config = load_config(run) if args.resume else Config()
    overrides = {key: getattr(args, key) for key in ('epochs', 'device', 'samples_per_class') if getattr(args, key) is not None}
    if args.subjects is not None:
        overrides['subjects'] = tuple(args.subjects)
    config = replace(config, **overrides)
    print(f'Run directory: {run.resolve()}', flush=True)
    run_training(args.original_dir, run, config, resume=args.resume,
                 stop_after_epoch=args.stop_after_epoch, generate=args.generate)


if __name__ == '__main__':
    main()
