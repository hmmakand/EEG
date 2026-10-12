"""Band-pass filtering, the analysed window, and per-subject Euclidean alignment."""
from dataclasses import replace

import numpy as np
from scipy import signal

from .config import DataConfig
from .recordings import Trial


def bandpass(data, band, sample_rate, order, time_axis=0):
    sos = signal.butter(order, band, "bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, data, axis=time_axis)


def analysed_signal(trial_signal, config: DataConfig):
    """[trial samples, channels] -> [window samples, channels].

    The whole trial is filtered first and cropped afterwards, so the filter's
    edge effects fall outside the window.
    """
    start, stop = config.window_samples
    return bandpass(trial_signal, config.band, config.sample_rate, config.filter_order)[start:stop]


def alignment_matrix(trials: list[Trial], config: DataConfig):
    """Inverse square root of the subject's mean spatial covariance (He & Wu, 2020)."""
    covariances = []
    for trial in trials:
        window = analysed_signal(trial.signal, config)
        covariances.append(window.T @ window / len(window))
    values, vectors = np.linalg.eigh(np.mean(covariances, axis=0))
    if values.min() <= 0:
        raise ValueError("Alignment needs a positive-definite mean covariance")
    return (vectors / np.sqrt(values)) @ vectors.T


def align_subject(trials: list[Trial], config: DataConfig) -> list[Trial]:
    """Whiten all trials of ONE subject with that subject's own matrix.

    Afterwards the subject's mean covariance in the analysed window is the
    identity. No labels are used. Filtering and cropping are linear in the
    channels, so aligning the raw trial equals aligning the analysed signal.
    """
    matrix = alignment_matrix(trials, config)
    return [replace(trial, signal=trial.signal @ matrix) for trial in trials]
