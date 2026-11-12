"""Band-pass filtering, the analysed stretch, and per-subject Euclidean alignment."""
from dataclasses import replace

import numpy as np
from scipy import signal

from .config import DataConfig
from .recording import Sample


def bandpass(data, band, sample_rate, order, time_axis=0):
    sos = signal.butter(order, band, "bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, data, axis=time_axis)


def analysed_signal(sample: Sample, config: DataConfig):
    """The band-passed stretch of one sample, [window samples, channels].

    The whole epoch is filtered first and cropped afterwards, so the filter's
    edge effects fall outside the stretch whenever the epoch has signal on
    both sides of it.
    """
    filtered = bandpass(sample.signal, config.band, config.sample_rate, config.filter_order)
    return filtered[sample.start:sample.start + config.window_samples]


RANK_TOLERANCE = 1e-10    # eigenvalues below this share of the largest count as absent


def alignment_matrix(samples: list[Sample], config: DataConfig):
    """Inverse square root of the subject's mean spatial covariance (He & Wu, 2020)."""
    covariances = []
    for sample in samples:
        window = analysed_signal(sample, config)
        covariances.append(window.T @ window / len(window))
    values, vectors = np.linalg.eigh(np.mean(covariances, axis=0))
    # Directions that a recording lacks hold no signal and are set to zero instead of being amplified.
    present = values > values.max() * RANK_TOLERANCE
    if not present.any():
        raise ValueError("Alignment needs a non-zero mean covariance")
    return (vectors / np.sqrt(np.where(present, values, np.inf))) @ vectors.T


def align_subject(samples: list[Sample], config: DataConfig) -> list[Sample]:
    """Whiten all samples of ONE subject with that subject's own matrix.

    Afterwards the subject's mean covariance over the analysed stretches is the
    identity. No labels are used. Filtering and cropping are linear in the
    channels, so aligning the epoch equals aligning the analysed signal.
    """
    matrix = alignment_matrix(samples, config)
    return [replace(sample, signal=sample.signal @ matrix) for sample in samples]
