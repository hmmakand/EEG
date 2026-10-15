"""Synthetic recordings of both datasets."""
from unittest import mock

import numpy as np
import scipy.io as sio

from ..datascripts import bci2a_config, liu2024_config
from ..datascripts.bci2a import subject_file as bci2a_file

SUBJECTS = (1, 2, 3)
CONFIG = liu2024_config(subjects=SUBJECTS)
BCI2A_CONFIG = bci2a_config(subjects=SUBJECTS, session_indices=(1, 2))   # session 0 also has trials; it is left out
ONSETS = (1004, 1, 1010, 1830, 1006)     # samples at 500 Hz: usual, at the epoch start, usual, too late, usual


def recording(onsets=ONSETS, seed=0, scale=1.0):
    """(EEG [epochs * 4000, 29], marker) with markers 1 (epoch start), 2 (imagery) and 3 (its end)."""
    rng = np.random.default_rng(seed)
    eeg = rng.normal(size=(4000 * len(onsets), 29)) * scale
    marker = np.zeros(len(eeg), dtype=int)
    for trial, onset in enumerate(onsets):
        start = 4000 * trial
        marker[start] = 1
        marker[start + onset] = 2
        marker[start + onset + 2000] = 3
    return eeg, marker


def fake_recordings():
    """Replace the EDF reader: every path gives a recording that depends on the subject number in its name."""
    def read(path):
        subject = int(str(path).split("sub-")[1][:2])
        return recording(seed=100 + subject, scale=1.0 + subject)
    return mock.patch("src.ourexpv27.datascripts.liu2024.read_recording", side_effect=read)


def write_bci2a_recordings(folder, subjects=SUBJECTS):
    """BCI IV 2a files: three sessions per subject; sessions 1 and 2 hold left, right, feet and one cut-off trial."""
    for subject in subjects:
        rng = np.random.default_rng(100 + subject)
        sessions = np.empty((1, 3), dtype=object)
        for index in range(3):
            signal = rng.normal(size=(6200, 25)) * (1 + subject)
            positions = np.array([100, 1200, 2300, 3400, 4500, 5600])  # the last trial does not fit
            labels = np.array([1, 2, 3, 2, 1, 2])
            record = np.empty((1, 1), dtype=[("X", object), ("trial", object), ("y", object)])
            record[0, 0] = (signal, positions.reshape(-1, 1), labels.reshape(-1, 1))
            sessions[0, index] = record
        sio.savemat(bci2a_file(folder, subject), {"data": sessions})
