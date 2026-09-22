"""All model, training, internal-split, and generation settings."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math

EXPERIMENT = Path(__file__).resolve().parents[2]
ORIGINAL = EXPERIMENT / 'datasyn' / 'original'
SYNTHETIC = EXPERIMENT / 'datasyn' / 'synthetic'


@dataclass(frozen=True)
class Config:
    subjects: tuple[int, ...] = (1, 2, 3, 5, 6, 7, 8, 9)
    split_seed: int = 42
    seed: int = 8888
    generation_seed: int = 8888
    validation_per_class: int = 5
    epochs: int = 1000
    batch_size: int = 1
    checkpoint_interval: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    min_learning_rate: float = 1e-6
    ema_decay: float = 0.9999
    signal_scale: float = 100.0
    in_channels: int = 22
    time_points: int = 1000
    num_classes: int = 2
    embed_dim: int = 512
    patch_size: int = 5
    depth: int = 4
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    learn_sigma: bool = True
    diffusion_steps: int = 1000
    noise_schedule: str = 'linear'
    samples_per_class: int = 50
    sample_batch_size: int = 1
    cfg_scale: float = 4.0
    device: str = 'cuda'

    def model_kwargs(self) -> dict:
        names = ('in_channels', 'time_points', 'num_classes', 'embed_dim', 'patch_size',
                 'depth', 'num_heads', 'mlp_ratio', 'class_dropout_prob', 'learn_sigma')
        return {name: getattr(self, name) for name in names}

    def to_dict(self) -> dict:
        result = asdict(self)
        result['subjects'] = list(self.subjects)
        return result

    @classmethod
    def from_dict(cls, values: dict) -> 'Config':
        return cls(**{**values, 'subjects': tuple(values['subjects'])})

    def validate(self) -> None:
        for name in ('epochs', 'batch_size', 'checkpoint_interval', 'validation_per_class', 'in_channels', 'time_points',
                     'embed_dim', 'patch_size', 'depth', 'num_heads', 'diffusion_steps',
                     'samples_per_class', 'sample_batch_size'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not self.subjects or len(set(self.subjects)) != len(self.subjects) or any(s < 1 for s in self.subjects):
            raise ValueError('Subjects must be unique positive IDs')
        if self.num_classes != 2 or not self.learn_sigma or self.noise_schedule != 'linear':
            raise ValueError('This adaptation uses two classes, learned variance, and linear noise')
        if self.time_points % self.patch_size or self.embed_dim % self.num_heads:
            raise ValueError('Time points must divide into patches; embedding dimension must divide into heads')
        if self.diffusion_steps < 20:
            raise ValueError('The reference linear beta schedule requires at least 20 steps')
        for name in ('split_seed', 'seed', 'generation_seed'):
            if not 0 <= getattr(self, name) < 2**32:
                raise ValueError(f'{name} must be in [0, 2**32)')
        if self.device != 'cpu' and self.device != 'cuda' and not self.device.startswith('cuda:'):
            raise ValueError('Device must be cpu, cuda, or cuda:N')
        for name in ('learning_rate', 'min_learning_rate', 'signal_scale', 'mlp_ratio'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not 0 <= self.weight_decay or not math.isfinite(self.weight_decay):
            raise ValueError('Weight decay must be finite and nonnegative')
        if not 0 < self.class_dropout_prob < 1 or not 0 <= self.ema_decay < 1:
            raise ValueError('Invalid label dropout or EMA decay')
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0:
            raise ValueError('Guidance scale must be finite and nonnegative')
