"""Local provenance, checkpoint, and reproducibility helpers."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .config import Config
from .diffusion import create_diffusion
from .models.eegdiffuser import EEGDiffuser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def make_model(config: Config):
    return EEGDiffuser(**config.model_kwargs()).to(config.device)


def make_diffusion(config: Config):
    return create_diffusion(timestep_respacing='', noise_schedule=config.noise_schedule,
                            diffusion_steps=config.diffusion_steps, learn_sigma=config.learn_sigma)


def load_config(run: Path) -> Config:
    config = Config.from_dict(json.loads((run / 'config.json').read_text()))
    config.validate()
    return config
