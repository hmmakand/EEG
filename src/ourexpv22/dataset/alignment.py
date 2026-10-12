"""Per-subject Euclidean alignment of trials (He & Wu, 2020).

Every trial of one subject is multiplied by R = mean_covariance^(-1/2), where
the covariance is taken over the analysed (filtered, windowed) part of that
subject's trials. Afterwards the subject's mean spatial covariance is the
identity, which removes subject-specific scale and channel mixing. Labels are
not used.
"""
from dataclasses import replace
import numpy as np
from .preprocessing import preprocess_trial


def alignment_matrix(trials, sample_rate, config):
    """R [channels, channels] for the trials of one subject."""
    covariances = []
    for trial in trials:
        signal = preprocess_trial(trial, sample_rate, config).signal
        covariances.append(signal.T @ signal / len(signal))
    values, vectors = np.linalg.eigh(np.mean(covariances, axis=0))
    if values.min() <= 0:
        raise ValueError("Alignment needs a positive-definite mean covariance")
    return (vectors / np.sqrt(values)) @ vectors.T


def align_trials(trials, sample_rate, config):
    """Return the trials of ONE subject after the configured alignment."""
    if config.alignment == "none" or not trials:
        return list(trials)
    if config.alignment != "euclidean":
        raise ValueError(f"Unknown alignment: {config.alignment}")
    # Filtering and windowing are linear in the channels, so aligning the raw
    # trial is the same as aligning the analysed signal.
    matrix = alignment_matrix(trials, sample_rate, config)
    return [replace(trial, signal=trial.signal @ matrix) for trial in trials]
