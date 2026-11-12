"""Read Liu2024 EDF recordings as full 8-second left/right imagery trials.

The published events.tsv assigns trial_type 1 (left) to the first trial and
trial_type 2 (right) to the next, alternating throughout the recording.
https://doi.org/10.1038/s41597-023-02787-8
The EDF marker channel encodes trial phases, not hand identity. Preserve the
original trial order when mapping even zero-based trials to left (0) and odd
trials to right (1). Each trial includes instruction, imagery, and rest.
"""
from pathlib import Path

import mne
import numpy as np

from .config import DataConfig
from .recording import Sample

EEG_CHANNELS = ("FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3", "C4",
                "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2")
MARKER_CHANNEL = ""                  # the EDF gives the marker channel an empty name
SOURCE_RATE = 500
EPOCH_START = 1                     # trial-start marker
MARKER_SCALE = 1e6                   # the reader returns the marker in volts; it was stored in microvolts
LEFT, RIGHT = 0, 1                  # labels


def subject_file(data_dir, subject: int) -> Path:
    """Accepts the MOABB folder (MNE-liu2024-data) or its `files` subfolder."""
    data_dir = Path(data_dir)
    root = data_dir / "files" if (data_dir / "files").is_dir() else data_dir
    name = f"sub-{subject:02d}"
    return root / "edffile" / name / "eeg" / f"{name}_task-motor-imagery_eeg.edf"


def read_recording(path):
    """One EDF -> (EEG [samples, 29] in volts, marker [samples] as integers)."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    if raw.info["sfreq"] != SOURCE_RATE:
        raise ValueError(f"{path}: expected {SOURCE_RATE} Hz, found {raw.info['sfreq']}")
    missing = [name for name in (*EEG_CHANNELS, MARKER_CHANNEL) if name not in raw.ch_names]
    if missing:
        raise ValueError(f"{path}: missing channels {missing}")
    eeg = raw.get_data(picks=list(EEG_CHANNELS)).T
    marker = np.rint(raw.get_data(picks=[MARKER_CHANNEL])[0] * MARKER_SCALE).astype(int)
    return eeg, marker


def cut_samples(eeg, marker, subject: int, config: DataConfig) -> list[Sample]:
    """Return one complete trial per sample, with alternating left/right labels."""
    if config.trial_seconds != 8.0 or config.sample_seconds != 8.0:
        raise ValueError("Liu2024 left/right decoding requires full 8-second trials and windows")
    step, remainder = divmod(SOURCE_RATE, config.sample_rate)
    epoch = config.trial_samples * step
    if remainder or not len(eeg) or len(eeg) % epoch:
        raise ValueError(f"Subject {subject}: recording must contain complete 8-second trials "
                         "and the sample rate must divide 500 Hz")
    if eeg.ndim != 2 or eeg.shape[1] != config.num_channels or marker.shape != (len(eeg),):
        raise ValueError("EEG channels or marker length do not match the recording configuration")
    samples = []
    for trial_id in range(len(eeg) // epoch):
        begin = trial_id * epoch
        markers = marker[begin:begin + epoch]
        if markers[0] != EPOCH_START:
            raise ValueError(f"Subject {subject}, trial {trial_id}: epoch-start marker not found")
        signal = eeg[begin:begin + epoch:step]
        label = LEFT if trial_id % 2 == 0 else RIGHT
        samples.append(Sample(signal, 0, label, subject, trial_id, trial_id))
    return samples


def read_samples(data_dir, subject: int, config: DataConfig) -> list[Sample]:
    """Full left/right trials of one subject, in recording order."""
    eeg, marker = read_recording(subject_file(data_dir, subject))
    return cut_samples(eeg, marker, subject, config)
