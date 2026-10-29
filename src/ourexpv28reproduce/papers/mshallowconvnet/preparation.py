"""Samples as M-ShallowConvNet receives them: [samples, channels, time] at 250 Hz, per subject.

Paper, section II-B: "EEG signals were low-pass filtered at 38 Hz using a
fourth-order Butterworth filter. In addition, we applied the exponential
moving standardization". The code (dataloader/bci_compet.py) filters the
continuous recording, converts to microvolts, cuts the epochs and
standardises every epoch on its own (braindecode, initial block of 4 s).

Here the filter runs over each trial's whole stretch (our recordings are
published as epochs), the sample is cut out, and the sample is standardised
on its own, as the code standardises its epochs. The filter is zero-phase,
as MNE's filter in the code is. The samples are those of version 27
(datascripts/version27.py).
"""
import numpy as np
from scipy import signal

from ...datascripts import SAMPLE_RATE
from ...datascripts.standardize import exponential_moving_standardize
from ...datascripts.version27 import read_samples
from .settings import PreparationSettings

MICROVOLTS = 1e6


def prepare_samples(samples, settings: PreparationSettings = PreparationSettings()):
    """(x [samples, channels, time] float32, labels, groups). Samples of one trial share its filtered stretch."""
    sos = signal.butter(settings.filter_order, settings.low_pass, "lowpass", fs=SAMPLE_RATE, output="sos")
    filtered, x = {}, []
    for sample in samples:
        key = id(sample.trial)
        if key not in filtered:
            filtered[key] = signal.sosfiltfilt(sos, sample.trial.signal * MICROVOLTS, axis=0)
        window = filtered[key][sample.start:sample.start + sample.length]
        x.append(exponential_moving_standardize(window, settings.standardize_factor, settings.standardize_init_block,
                                                settings.standardize_eps).T)
    return (np.stack(x).astype(np.float32), np.array([s.label for s in samples], dtype=np.int64),
            np.array([s.group for s in samples]))


def load_subject(data_dir, dataset: str, subject: int, settings: PreparationSettings = PreparationSettings()):
    """(x, labels, groups) of one subject: version 27's samples, prepared as M-ShallowConvNet's paper."""
    return prepare_samples(read_samples(data_dir, dataset, subject), settings)
