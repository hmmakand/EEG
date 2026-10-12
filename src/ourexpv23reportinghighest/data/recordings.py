"""Read BCI Competition IV 2a MATLAB files and cut out the left/right trials."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from .config import DataConfig


@dataclass(frozen=True)
class Trial:
    signal: np.ndarray  # [samples, channels]
    label: int          # 0 left hand, 1 right hand
    subject_id: int
    session_id: int
    trial_id: int       # index of the event inside its session


def subject_file(data_dir, subject: int) -> Path:
    return Path(data_dir) / f"A{subject:02d}T.mat"


def read_trials(data_dir, subject: int, config: DataConfig) -> list[Trial]:
    """All complete left/right trials of one subject, ordered by session and event."""
    data = sio.loadmat(subject_file(data_dir, subject))["data"]
    labels = dict(config.label_map)
    trials = []
    sessions = range(data.shape[1]) if config.session_indices is None else config.session_indices
    for session_id in sessions:
        if session_id >= data.shape[1]:
            raise ValueError(f"Subject {subject}: missing session index {session_id}")
        record = data[0, session_id][0, 0]
        signal, positions, classes = record[0], record[1].flatten(), record[2].flatten()
        if len(positions) == 0:
            continue  # a run without trials (eye-movement recording)
        signal = signal[:, list(config.channel_indices)]
        for trial_id, (position, source_label) in enumerate(zip(positions, classes)):
            label = labels.get(int(source_label))
            if label is None:
                continue  # feet or tongue
            start = int(position)
            stop = start + config.trial_samples
            if start < 0 or stop > len(signal):
                continue  # incomplete trial
            trials.append(Trial(signal[start:stop], label, subject, session_id, trial_id))
    if not trials:
        raise ValueError(f"No left/right trials found for subject {subject}")
    return trials
