"""Read BCI Competition IV 2a MATLAB files and cut out the left/right trials."""
from pathlib import Path

import scipy.io as sio

from .config import DataConfig
from .recordings import Trial

LABELS = {1: 0, 2: 1}       # left hand -> 0, right hand -> 1; feet (3) and tongue (4) are not used
CUE_SECONDS = 2.0           # an event marks the trial start; the cue follows 2 s later


def subject_file(data_dir, subject: int) -> Path:
    return Path(data_dir) / f"A{subject:02d}T.mat"


def read_trials(data_dir, subject: int, config: DataConfig) -> list[Trial]:
    """All complete left/right trials of one subject, ordered by session and event."""
    onset = int(CUE_SECONDS * config.sample_rate)
    data = sio.loadmat(subject_file(data_dir, subject))["data"]
    trials = []
    sessions = range(data.shape[1]) if config.session_indices is None else config.session_indices
    for session_id in sessions:
        if session_id >= data.shape[1]:
            raise ValueError(f"Subject {subject}: missing session index {session_id}")
        record = data[0, session_id][0, 0]
        signal, positions, classes = record[0], record[1].flatten(), record[2].flatten()
        if len(positions) == 0:
            continue  # a run without trials (eye-movement recording)
        signal = signal[:, :config.num_channels]   # the EEG electrodes come first
        for trial_id, (position, source_label) in enumerate(zip(positions, classes)):
            label = LABELS.get(int(source_label))
            if label is None:
                continue  # feet or tongue
            start = int(position)
            stop = start + config.trial_samples
            if start < 0 or stop > len(signal):
                continue  # incomplete trial
            trials.append(Trial(signal[start:stop], onset, label, subject, session_id, trial_id))
    return trials
