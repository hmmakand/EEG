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

from .trial import SAMPLE_RATE, Trial

SOURCE_RATE = 512
BLOCKS = ("DATA1", "DATA2")
FILES = (("left", 0), ("right", 1))      # file name part, label
MICROVOLT = 1e-6
NUM_CHANNELS = 63                        # 10-10 system; the files carry no channel names
SUBJECTS = tuple(range(1, 16))
TRIAL_SECONDS = 6.832                    # 1,708 samples at 250 Hz; the shortest published epoch is 3,500 at 512 Hz
CUE_SECONDS = 3.0                        # estimated, see the module text


def class_file(data_dir, subject: int, side: str) -> Path:
    return Path(data_dir) / f"exp1-S{subject}-{side}.mat"


def subject_file(data_dir, subject: int) -> Path:
    """The left-hand file; the right-hand file lies next to it."""
    return class_file(data_dir, subject, "left")


def resample(epoch):
    """[channels, source samples] at 512 Hz -> [trial samples, channels] at the configured rate, in volts.

    Only the first 3,500 source samples are used, so that every subject gives
    samples of the same length.
    """
    source_samples = int(round(TRIAL_SECONDS * SOURCE_RATE))
    if epoch.shape[1] < source_samples:
        raise ValueError(f"An epoch of {epoch.shape[1]} samples is shorter than trial_seconds")
    divisor = np.gcd(SAMPLE_RATE, SOURCE_RATE)
    resampled = resample_poly(epoch[:, :source_samples], SAMPLE_RATE // divisor, SOURCE_RATE // divisor,
                              axis=1, padtype="line")
    return np.ascontiguousarray(resampled[:, :int(TRIAL_SECONDS * SAMPLE_RATE)].T) * MICROVOLT


def read_trials(data_dir, subject: int) -> list[Trial]:
    """The 80 trials of one subject, one sample each: block 1 left, block 1 right, block 2 left, block 2 right.

    `onset` is the cue time estimated from the recordings (CUE_SECONDS).
    """
    onset = int(round(CUE_SECONDS * SAMPLE_RATE))
    files = {}
    for side, label in FILES:
        path = class_file(data_dir, subject, side)
        if not path.is_file():
            raise ValueError(f"Missing recording {path}")
        files[label] = sio.loadmat(path)
    samples = []
    for key in BLOCKS:
        for label in (0, 1):
            for epoch in files[label][key][0]:
                epoch = np.asarray(epoch, dtype=float)
                if epoch.ndim != 2 or epoch.shape[0] != NUM_CHANNELS:
                    raise ValueError(f"Subject {subject}, {key}: expected {NUM_CHANNELS} channels, "
                                     f"found an array of shape {epoch.shape}")
                samples.append(Trial(resample(epoch), onset, label, subject, len(samples)))
    return samples
