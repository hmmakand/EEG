"""Samples as MSCFormer receives them: [samples, channels, time] at 250 Hz, per subject.

Paper, "Data preprocessing": "we did not apply any band-pass filtering or
artifact removal techniques", only a zero-mean standardisation with the mean
and standard deviation of the training data, applied to the test data (one
value each, as the code computes them over all training values). The samples
are version 27's (datascripts/version27.py); the standardisation is applied
per fold in training.py.
"""
import numpy as np

from ...datascripts.version27 import read_samples


def prepare_samples(samples):
    """(x [samples, channels, time] float32 in volts, unfiltered; labels; groups)."""
    x = [s.trial.signal[s.start:s.start + s.length].T for s in samples]
    return (np.stack(x).astype(np.float32), np.array([s.label for s in samples], dtype=np.int64),
            np.array([s.group for s in samples]))


def load_subject(data_dir, dataset: str, subject: int):
    return prepare_samples(read_samples(data_dir, dataset, subject))


def z_score(train, *others):
    """Scale every array with the mean and standard deviation of all values of `train`."""
    mean, std = float(np.mean(train)), float(np.std(train))
    return [((a - mean) / std).astype(np.float32) for a in (train, *others)]
