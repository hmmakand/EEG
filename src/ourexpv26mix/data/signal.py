"""Band-pass filtering, the analysed window, and per-subject Euclidean alignment."""
from dataclasses import replace

import numpy as np
from scipy import signal

from .config import DataConfig
from .recordings import Trial


def bandpass(data, band, sample_rate, order, time_axis=0):
    sos = signal.butter(order, band, "bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, data, axis=time_axis)


def analysed_signal(trial: Trial, config: DataConfig, low: bool = False):
    """The band-passed window of one trial, [window samples, channels].

    The whole trial is filtered first and cropped afterwards, so the filter's
    edge effects fall outside the window whenever the trial has signal on
    both sides of it. The window is placed at the trial's own imagery onset.
    `low` selects the low-frequency band (config.low_band) instead of the main band.
    """
    start, stop = (trial.onset + offset for offset in config.window_offsets)
    band, order = (config.low_band, config.low_filter_order) if low else (config.band, config.filter_order)
    return bandpass(trial.signal, band, config.sample_rate, order)[start:stop]


def low_frequency_signal(trial: Trial, config: DataConfig):
    """The low-band window at config.low_sample_rate, each channel with its mean removed: [low samples, channels].

    `trial` must be aligned in the low band (align_subject(..., low=True)).
    """
    window = analysed_signal(trial, config, low=True)[::config.sample_rate // config.low_sample_rate]
    return window - window.mean(axis=0, keepdims=True)


RANK_TOLERANCE = 1e-10    # eigenvalues below this share of the largest count as absent


def alignment_matrix(trials: list[Trial], config: DataConfig, low: bool = False):
    """Inverse square root of the subject's mean spatial covariance (He & Wu, 2020), in the main or the low band."""
    covariances = []
    for trial in trials:
        window = analysed_signal(trial, config, low)
        covariances.append(window.T @ window / len(window))
    values, vectors = np.linalg.eigh(np.mean(covariances, axis=0))
    # A recording can lack one or more dimensions (for example after re-referencing:
    # jia2019 subject 6 has rank 62 of 63). Those directions hold no signal and are
    # set to zero instead of being amplified.
    present = values > values.max() * RANK_TOLERANCE
    if not present.any():
        raise ValueError("Alignment needs a non-zero mean covariance")
    return (vectors / np.sqrt(np.where(present, values, np.inf))) @ vectors.T


def align_subject(trials: list[Trial], config: DataConfig, low: bool = False) -> list[Trial]:
    """Whiten all trials of ONE subject with that subject's own matrix.

    Afterwards the subject's mean covariance in the analysed window is the
    identity (on the dimensions the recording has). No labels are used. Filtering and cropping are linear in the
    channels, so aligning the raw trial equals aligning the analysed signal.
    """
    matrix = alignment_matrix(trials, config, low)
    return [replace(trial, signal=trial.signal @ matrix) for trial in trials]
