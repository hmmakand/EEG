"""Deterministic MATLAB fixture matching the loader's nested data structure."""
from pathlib import Path
import numpy as np
from scipy.io import savemat


def write_subject(directory):
    rng = np.random.default_rng(2026)
    sessions = np.empty((1, 9), dtype=object)
    for index in range(9):
        record = np.empty((1, 1), dtype=[('X', 'O'), ('trial', 'O'), ('y', 'O')])
        record[0, 0]['X'] = rng.standard_normal((2100, 25))
        record[0, 0]['trial'] = np.array([[0], [1000]])
        record[0, 0]['y'] = np.array([[1], [2]])
        sessions[0, index] = record
    savemat(Path(directory) / 'A01T.mat', {'data': sessions})
