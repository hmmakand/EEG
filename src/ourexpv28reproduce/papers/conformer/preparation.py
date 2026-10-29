"""Samples as EEG Conformer receives them: [samples, channels, time] at 250 Hz, per subject.

Paper, section II-B: a 6th-order Chebyshev band-pass, then "a Z-score
standardization [...] mean and variance, calculated with the training data
and used directly for the test data." preprocessing/BCIIV2a.m filters every
cut-out trial with cheby2(6, 60, [4 40] Hz) and filtfilt; conformer.py takes
one mean and one standard deviation over all training values.

Here every sample (version 27's window) is filtered on its own, as the
MATLAB code filters its cut-out trials, with second-order sections for
numerical stability. The z-score is applied in training.py, per fold, from
that fold's training samples.
"""
import numpy as np
from scipy import signal

from ...datascripts import SAMPLE_RATE
from ...datascripts.version27 import read_samples
from .settings import PreparationSettings


def prepare_samples(samples, settings: PreparationSettings = PreparationSettings()):
    """(x [samples, channels, time] float32 in volts, filtered; labels; groups)."""
    sos = signal.cheby2(settings.order, settings.stopband_db, settings.band, "bandpass", fs=SAMPLE_RATE,
                        output="sos")
    x = [signal.sosfiltfilt(sos, s.trial.signal[s.start:s.start + s.length], axis=0).T for s in samples]
    return (np.stack(x).astype(np.float32), np.array([s.label for s in samples], dtype=np.int64),
            np.array([s.group for s in samples]))


def load_subject(data_dir, dataset: str, subject: int, settings: PreparationSettings = PreparationSettings()):
    """(x, labels, groups) of one subject: version 27's samples, filtered as EEG Conformer's code."""
    return prepare_samples(read_samples(data_dir, dataset, subject), settings)


def z_score(train, *others):
    """Scale every array with the mean and standard deviation of all values of `train` (conformer.py)."""
    mean, std = float(np.mean(train)), float(np.std(train))
    return [(a - mean) / std for a in (train, *others)]
