"""A trial, and the choice of reader for the configured dataset."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DataConfig


@dataclass(frozen=True)
class Trial:
    signal: np.ndarray  # [samples, channels]
    onset: int          # sample of `signal` at which the motor imagery starts
    label: int          # 0 left hand, 1 right hand
    subject_id: int
    session_id: int
    trial_id: int       # index of the trial inside its session


def _reader(config: DataConfig):
    from . import bci2a, jia2019, liu2024   # the readers import Trial from this module
    return dict(liu2024=liu2024, jia2019=jia2019, bci2a=bci2a)[config.dataset]


def subject_file(data_dir, subject: int, config: DataConfig) -> Path:
    return _reader(config).subject_file(data_dir, subject)


def read_trials(data_dir, subject: int, config: DataConfig) -> list[Trial]:
    """All complete left/right trials of one subject, in recording order."""
    trials = _reader(config).read_trials(data_dir, subject, config)
    if not trials:
        raise ValueError(f"No left/right trials found for subject {subject}")
    for trial in trials:
        start, stop = (trial.onset + offset for offset in config.window_offsets)
        if trial.signal.shape[1] != config.num_channels or start < 0 or stop > len(trial.signal):
            raise ValueError(f"Subject {subject}, trial {trial.trial_id}: the analysed window does not fit "
                             f"({config.num_channels} channels, samples {start}-{stop} of {trial.signal.shape})")
    return trials
