"""Samples as SlimSeiz receives them: [samples, channels, time] at 250 Hz, per subject.

The code reads the raw EDF signals (volts), multiplies them by 1e4 and feeds
them to the network without filtering or normalisation (Loop_ADDNet_kFold.ipynb:
`data = data * 1e4`). Here the samples are version 27's (datascripts/version27.py);
the readers return volts, except the BCI IV 2a reader, which returns
microvolts and is converted to volts first, so that every dataset reaches the
network on the same scale.
"""
import numpy as np

from ...datascripts.version27 import read_samples

SCALE = 1e4
MICROVOLT_READERS = ("bci2a",)


def prepare_samples(samples, dataset: str):
    """(x [samples, channels, time] float32, volts x 1e4; labels; groups)."""
    to_volts = 1e-6 if dataset in MICROVOLT_READERS else 1.0
    x = [s.trial.signal[s.start:s.start + s.length].T * to_volts * SCALE for s in samples]
    return (np.stack(x).astype(np.float32), np.array([s.label for s in samples], dtype=np.int64),
            np.array([s.group for s in samples]))


def load_subject(data_dir, dataset: str, subject: int):
    return prepare_samples(read_samples(data_dir, dataset, subject), dataset)
