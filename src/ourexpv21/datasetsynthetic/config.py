"""Signal extraction and fold rules; no permanent train/validation/test split."""
from dataclasses import dataclass, field
from pathlib import Path

if __package__ == 'datasetsynthetic':
    from dataset.config import TrialConfig
else:
    from ..dataset.config import TrialConfig

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'datasyn' / 'fold_first'


@dataclass(frozen=True)
class PreparationConfig:
    subjects: tuple[int, ...] = (1, 2, 3, 5, 6, 7, 8, 9)
    seed: int = 42
    n_folds: int = 10
    sample_rate: int = 250
    trials: TrialConfig = field(default_factory=lambda: TrialConfig(order='grouped'))
    samples_per_class: int = 50
    validation_per_class: int = 5

    def validate(self):
        if len(self.subjects) < 2 or len(set(self.subjects)) != len(self.subjects):
            raise ValueError('Both protocols require at least two unique subjects')
        if any(type(s) is not int or s < 1 for s in self.subjects):
            raise ValueError('Subjects must be positive integers')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Seed must be an integer in [0, 2**32)')
        for name in ('n_folds', 'sample_rate', 'samples_per_class', 'validation_per_class'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.n_folds < 2:
            raise ValueError('At least two within-subject folds are required')
        if self.trials.duration_seconds <= 0 or self.trials.label_map != ((1, 0), (2, 1)):
            raise ValueError('Expected positive duration and binary left/right labels')
        for values in (self.trials.session_indices, self.trials.channel_indices):
            if not values or len(set(values)) != len(values) or min(values) < 0:
                raise ValueError('Session/channel indices must be unique and nonnegative')
