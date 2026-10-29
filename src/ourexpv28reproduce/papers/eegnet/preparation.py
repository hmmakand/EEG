"""Samples as EEGNet receives them: [samples, channels, time] at 128 Hz, per subject.

Paper, section 2.1.4 (the SMR data, BCI IV 2a): "We resampled the timeseries
to 128 Hz, and follow the same EEG pre-processing procedure as described in
[32] (Schirrmeister et al. 2017), using software that was provided by the
authors." Table 1 gives the band, 4-40 Hz. The software of [32] (braindecode)
converts to microvolts, band-passes with a causal Butterworth filter and then
applies exponential moving standardisation per channel.

[32]'s code works on continuous recordings; ours are trials (Liu 2024 and Jia
2019 are published as separate epochs). Each trial's whole stretch is
therefore filtered and standardised on its own, and the sample is cut from it
afterwards. The stretches are those of version 27 (datascripts/version27.py).
"""
import numpy as np
from scipy import signal

from ...datascripts import SAMPLE_RATE
from ...datascripts.standardize import exponential_moving_standardize
from ...datascripts.version27 import read_samples
from .settings import PreparationSettings

MICROVOLTS = 1e6


def prepare_stretch(stretch, settings: PreparationSettings):
    """One trial's stretch [time, channels] in volts at 250 Hz -> microvolts, filtered, standardised, 128 Hz."""
    sos = signal.butter(settings.filter_order, settings.band, "bandpass", fs=SAMPLE_RATE, output="sos")
    filtered = signal.sosfilt(sos, stretch * MICROVOLTS, axis=0)                       # causal, as braindecode
    standardized = exponential_moving_standardize(filtered, settings.standardize_factor,
                                                  settings.standardize_init_block, settings.standardize_eps)
    divisor = np.gcd(settings.sample_rate, SAMPLE_RATE)
    return signal.resample_poly(standardized, settings.sample_rate // divisor, SAMPLE_RATE // divisor, axis=0)


def prepare_samples(samples, settings: PreparationSettings = PreparationSettings()):
    """(x [samples, channels, time] float32, labels, groups). Samples of one trial share its prepared stretch."""
    scale = settings.sample_rate / SAMPLE_RATE
    prepared, x = {}, []
    for sample in samples:
        key = id(sample.trial)
        if key not in prepared:
            prepared[key] = prepare_stretch(sample.trial.signal, settings)
        stretch = prepared[key]
        start, length = int(round(sample.start * scale)), int(round(sample.length * scale))
        if start + length > len(stretch):
            raise ValueError(f"Subject {sample.trial.subject_id}, trial {sample.trial.trial_id}: the window does "
                             f"not fit at {settings.sample_rate} Hz")
        x.append(stretch[start:start + length].T)
    return (np.stack(x).astype(np.float32), np.array([s.label for s in samples], dtype=np.int64),
            np.array([s.group for s in samples]))


def load_subject(data_dir, dataset: str, subject: int, settings: PreparationSettings = PreparationSettings()):
    """(x, labels, groups) of one subject: version 27's samples, prepared as EEGNet's paper."""
    return prepare_samples(read_samples(data_dir, dataset, subject), settings)
