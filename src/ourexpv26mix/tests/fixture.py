"""Small synthetic recordings in the layout of the BCI Competition IV 2a MATLAB files."""
import numpy as np
import scipy.io as sio

from ..data import bci2a_config
from ..data.bci2a import subject_file

SUBJECTS = (1, 2, 3)
CONFIG = bci2a_config(subjects=SUBJECTS, session_indices=(1, 2))   # session 0 also has trials; it is left out


def write_recordings(folder, subjects=SUBJECTS):
    """Each subject: three sessions; sessions 1 and 2 hold left, right, feet and one cut-off trial."""
    for subject in subjects:
        rng = np.random.default_rng(100 + subject)
        sessions = np.empty((1, 3), dtype=object)
        for index in range(3):
            signal = rng.normal(size=(6200, 25)) * (1 + subject)
            positions = np.array([100, 1200, 2300, 3400, 4500, 5600])  # the last trial does not fit
            labels = np.array([1, 2, 3, 2, 1, 2])
            sessions[0, index] = _struct(signal, positions, labels)
        sio.savemat(subject_file(folder, subject), {"data": sessions})


def _struct(signal, positions, labels):
    """One session record: fields in the order signal, event positions, class labels."""
    record = np.empty((1, 1), dtype=[("X", object), ("trial", object), ("y", object)])
    record[0, 0] = (signal, positions.reshape(-1, 1), labels.reshape(-1, 1))
    return record
