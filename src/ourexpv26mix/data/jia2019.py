"""Read the Jia 2019 stroke-patient recordings.

Jia T., "EEG data of motor imagery for stroke", figshare (2019),
DOI 10.6084/m9.figshare.7636301. Fifteen stroke patients imagined moving the
left or the right hand.

Files, two per subject: `exp1-S<n>-left.mat` and `exp1-S<n>-right.mat`. The
class is given by the file. Each file holds two blocks, `DATA1` and `DATA2`,
of 20 trials; a trial is an array [63 channels, samples] in microvolts, with
3,500 samples for subjects 1-8 and 4,000 for subjects 9-15.

What the publication does not provide:
  - channel names (63 electrodes of the 10-10 system at 512 Hz are reported
    by Wang et al. 2026, arXiv:2604.16554; the order is unknown),
  - the time of the cue inside an epoch. It was estimated from the recordings
    without using the labels: averaged over all trials, the subjects show an
    evoked response that peaks 3.1 to 3.3 s into the epoch, followed by a drop
    of the 8-30 Hz power. The cue is therefore taken to be at 3.0 s
    (CUE_SECONDS), and that is the onset given to every trial,
  - the order in which the trials were recorded. Left-hand and right-hand
    trials come in separate files; whether they were recorded interleaved or
    in separate runs is not stated.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from .config import DataConfig
from .recordings import Trial

SOURCE_RATE = 512
BLOCKS = ("DATA1", "DATA2")
FILES = (("left", 0), ("right", 1))      # file name part, label
MICROVOLT = 1e-6
CUE_SECONDS = 3.0                        # estimated, see the module text


def class_file(data_dir, subject: int, side: str) -> Path:
    return Path(data_dir) / f"exp1-S{subject}-{side}.mat"


def subject_file(data_dir, subject: int) -> Path:
    """The left-hand file; the right-hand file lies next to it."""
    return class_file(data_dir, subject, "left")


def resample(epoch, config: DataConfig):
    """[channels, source samples] at 512 Hz -> [trial samples, channels] at the configured rate, in volts.

    Only the first 3,500 source samples are used, so that every subject gives
    trials of the same length.
    """
    source_samples = int(round(config.trial_seconds * SOURCE_RATE))
    if epoch.shape[1] < source_samples:
        raise ValueError(f"An epoch of {epoch.shape[1]} samples is shorter than trial_seconds")
    divisor = np.gcd(config.sample_rate, SOURCE_RATE)
    resampled = resample_poly(epoch[:, :source_samples], config.sample_rate // divisor, SOURCE_RATE // divisor,
                              axis=1, padtype="line")
    return np.ascontiguousarray(resampled[:, :config.trial_samples].T) * MICROVOLT


def read_trials(data_dir, subject: int, config: DataConfig) -> list[Trial]:
    """The 80 trials of one subject: block 1 left, block 1 right, block 2 left, block 2 right.

    `session_id` is the block and `trial_id` the position inside the block's file.
    """
    onset = int(round(CUE_SECONDS * config.sample_rate))
    files = {}
    for side, label in FILES:
        path = class_file(data_dir, subject, side)
        if not path.is_file():
            raise ValueError(f"Missing recording {path}")
        files[label] = sio.loadmat(path)
    trials = []
    for block, key in enumerate(BLOCKS):
        for label in (0, 1):
            for trial_id, epoch in enumerate(files[label][key][0]):
                epoch = np.asarray(epoch, dtype=float)
                if epoch.ndim != 2 or epoch.shape[0] != config.num_channels:
                    raise ValueError(f"Subject {subject}, {key}: expected {config.num_channels} channels, "
                                     f"found an array of shape {epoch.shape}")
                trials.append(Trial(resample(epoch, config), onset, label, subject, block, trial_id))
    return trials
