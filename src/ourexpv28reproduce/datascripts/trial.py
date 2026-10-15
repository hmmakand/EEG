"""A trial as every reader returns it."""
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 250   # every reader returns signals at this rate


@dataclass(frozen=True)
class Trial:
    signal: np.ndarray  # [samples, channels] at SAMPLE_RATE, in volts: the whole stretch that was read
    onset: int          # sample of `signal` at which the cue or the imagery starts
    label: int          # 0 left hand, 1 right hand
    subject_id: int
    trial_id: int       # index of the trial inside the subject, in recording order
