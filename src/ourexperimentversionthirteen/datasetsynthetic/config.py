"""Settings for original EEG extraction and fixed within-subject partitions."""
from dataclasses import dataclass, field
from pathlib import Path

if __package__ == 'datasetsynthetic':
    from dataset.config import TrialConfig
else:
    from ..dataset.config import TrialConfig


@dataclass(frozen=True)
class SplitConfig:
    subjects: tuple[int, ...] = (1, 2, 3, 5, 6, 7, 8, 9)
    seed: int = 42
    # Integer percentages allow exact largest-remainder rounding.
    percentages: tuple[int, int, int] = (70, 15, 15)
    sample_rate: int = 250
    trials: TrialConfig = field(default_factory=lambda: TrialConfig(order='grouped'))


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'datasyn' / 'original'
PARTITIONS = ('training', 'validation', 'testing')
