"""Samples as DBConformer receives them, and the Euclidean alignment of the code.

Section IV-A4: MOABB's standard pre-processing, then "EA was utilized after
pre-processing on all MI datasets, and the EA reference matrix was updated
online when new test trials arrived on-the-fly". In the code (utils/utils.py,
alg_utils.py):

  training trials  aligned per subject with the mean covariance of all its trials (EA)
  test trials      aligned one after another: trial k with the mean covariance of trials 1 to k (online EA)

The covariance is numpy's (mean removed). Each trial's whole stretch is
band-passed (8-32 Hz) and the sample cut from it; samples are version 27's.
Where a covariance is rank-deficient (jia2019 has a common average reference,
and subject 6 one channel too few), the inverse square root leaves out the
missing directions; for full-rank covariances it is the code's
fractional_matrix_power(R, -0.5).
"""
import numpy as np
from scipy import signal

from ...datascripts import SAMPLE_RATE
from ...datascripts.version27 import read_samples
from .settings import PreparationSettings

MICROVOLTS = 1e6


def prepare_samples(samples, settings: PreparationSettings = PreparationSettings()):
    """(x [samples, channels, time] float64 in microvolts, band-passed, not aligned; labels; groups)."""
    sos = signal.butter(settings.filter_order, settings.band, "bandpass", fs=SAMPLE_RATE, output="sos")
    filtered, x = {}, []
    for sample in samples:
        key = id(sample.trial)
        if key not in filtered:
            filtered[key] = signal.sosfiltfilt(sos, sample.trial.signal * MICROVOLTS, axis=0)
        x.append(filtered[key][sample.start:sample.start + sample.length].T)
    return np.stack(x), np.array([s.label for s in samples], dtype=np.int64), np.array([s.group for s in samples])


def load_subject(data_dir, dataset: str, subject: int, settings: PreparationSettings = PreparationSettings()):
    return prepare_samples(read_samples(data_dir, dataset, subject), settings)


def inverse_sqrt(matrix, tolerance: float):
    values, vectors = np.linalg.eigh(matrix)
    keep = values > tolerance * values.max()
    scale = np.zeros_like(values)
    scale[keep] = values[keep] ** -0.5
    return (vectors * scale) @ vectors.T


def covariances(x):
    return np.stack([np.cov(trial) for trial in x])


def align(x, tolerance: float = PreparationSettings().rank_tolerance):
    """EA (alg_utils.EA): every trial times the inverse square root of the trials' mean covariance."""
    reference = inverse_sqrt(covariances(x).mean(axis=0), tolerance)
    return np.einsum("ij,njt->nit", reference, x)


def align_online(x, tolerance: float = PreparationSettings().rank_tolerance):
    """Online EA (utils.data_loader): trial k with the mean covariance of trials 1..k, in the given order."""
    running = np.cumsum(covariances(x), axis=0) / np.arange(1, len(x) + 1)[:, None, None]
    return np.stack([inverse_sqrt(r, tolerance) @ trial for r, trial in zip(running, x)])
