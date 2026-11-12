"""Read BCI Competition IV 2a MATLAB files and cut out the left-hand and right-hand trials.

Files `A01T.mat` ... `A09T.mat`: a list of sessions, each with the signal
(22 EEG electrodes first, then 3 EOG channels, 250 Hz), the positions of the
trial starts and the class of every trial. The cue comes 2 s after the trial
start.
"""
from pathlib import Path

import scipy.io as sio

from .config import DataConfig
from .recording import Sample

EEG_CHANNELS = ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz")
LABELS = {1: 0, 2: 1}       # left hand -> 0, right hand -> 1; feet (3) and tongue (4) are not used


def subject_file(data_dir, subject: int) -> Path:
    return Path(data_dir) / f"A{subject:02d}T.mat"


def read_samples(data_dir, subject: int, config: DataConfig) -> list[Sample]:
    """One sample per complete left/right trial, ordered by session and event; the stretch starts at the cue."""
    cue = int(config.cue_seconds * config.sample_rate)
    data = sio.loadmat(subject_file(data_dir, subject))["data"]
    samples = []
    sessions = range(data.shape[1]) if config.session_indices is None else config.session_indices
    for session_id in sessions:
        if session_id >= data.shape[1]:
            raise ValueError(f"Subject {subject}: missing session index {session_id}")
        record = data[0, session_id][0, 0]
        signal, positions, classes = record[0], record[1].flatten(), record[2].flatten()
        if len(positions) == 0:
            continue  # a run without trials (eye-movement recording)
        signal = signal[:, :config.num_channels]   # the EEG electrodes come first
        for position, source_label in zip(positions, classes):
            label = LABELS.get(int(source_label))
            if label is None:
                continue  # feet or tongue
            start = int(position)
            stop = start + config.trial_samples
            if start < 0 or stop > len(signal):
                continue  # incomplete trial
            samples.append(Sample(signal[start:stop], cue, label, subject, len(samples), len(samples)))
    return samples
