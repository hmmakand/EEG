"""Read the Liu et al. 2024 stroke-patient recordings and cut out the left/right trials.

Liu H. et al., "An EEG motor imagery dataset for brain computer interface in
acute stroke patients", Scientific Data 11, 131 (2024). Files as downloaded
by MOABB: one EDF per subject and one events table shared by all subjects.

What the EDF of a subject holds (checked on all 50 subjects):

  - 160,000 samples at 500 Hz: 40 epochs of 8 s, concatenated. The epochs are
    not contiguous in time (the signal jumps where two epochs meet), so every
    epoch is handled on its own.
  - 33 channels: 29 EEG electrodes, the reference CPz (all zeros), two EOG
    channels and one marker channel.
  - The published signal is already band-passed at 0.5-40 Hz.
  - The marker channel is 1 at the first sample of every epoch, 2 where the
    motor imagery starts and 3 where it ends, 4 s later.

The imagery does not start at the same time in every epoch: 2.0 s after the
epoch start in 1,929 of the 2,000 trials, at the epoch start in 23, and
between 2.2 s and 3.7 s in 48. Each trial's own marker is therefore used.
"""
import csv
from pathlib import Path

import mne
import numpy as np

from .config import DataConfig
from .recordings import Trial

EEG_CHANNELS = ("FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3", "C4",
                "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2")
MARKER_CHANNEL = ""                  # the EDF gives the marker channel an empty name
SOURCE_RATE = 500
EPOCH_SECONDS = 8.0
EPOCH_START, IMAGERY_START = 1, 2    # marker values
MARKER_SCALE = 1e6                   # the reader returns the marker in volts; it was stored in microvolts
LABELS = {"1": 0, "2": 1}            # trial_type of the events table: left hand -> 0, right hand -> 1
EVENT_FILES = ("38516084", "task-motor-imagery_events.tsv")   # name given by MOABB, name in the publication


def _root(data_dir) -> Path:
    """Accept the MOABB folder (MNE-liu2024-data) or its `files` subfolder."""
    data_dir = Path(data_dir)
    return data_dir / "files" if (data_dir / "files").is_dir() else data_dir


def subject_file(data_dir, subject: int) -> Path:
    name = f"sub-{subject:02d}"
    return _root(data_dir) / "edffile" / name / "eeg" / f"{name}_task-motor-imagery_eeg.edf"


def read_labels(data_dir) -> list[int]:
    """Class of every trial, in order. The table lists three stages per trial; one label is kept per trial."""
    root = _root(data_dir)
    path = next((root / name for name in EVENT_FILES if (root / name).is_file()), None)
    if path is None:
        raise ValueError(f"No events table ({' or '.join(EVENT_FILES)}) in {root}")
    with open(path, newline="") as table:
        imagery_rows = [row for row in csv.DictReader(table, delimiter="\t") if row["value"] == "2"]
    return [LABELS[row["trial_type"]] for row in imagery_rows]


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


def cut_trials(eeg, marker, labels, subject: int, config: DataConfig) -> list[Trial]:
    """Split a recording into its epochs and locate the imagery onset of each.

    The published signal has no content above 40 Hz, so taking every second
    sample (500 Hz -> 250 Hz) loses nothing and needs no further filter.
    """
    step, remainder = divmod(SOURCE_RATE, config.sample_rate)
    epoch = int(EPOCH_SECONDS * SOURCE_RATE)
    if remainder or config.trial_samples * step != epoch:
        raise ValueError("liu2024 needs trial_seconds = 8 and a sample rate that divides 500 Hz")
    if len(eeg) != len(labels) * epoch:
        raise ValueError(f"Subject {subject}: {len(eeg)} samples do not hold {len(labels)} epochs of {epoch}")
    trials = []
    for trial_id, label in enumerate(labels):
        start = trial_id * epoch
        markers = marker[start:start + epoch]
        imagery = np.flatnonzero(markers == IMAGERY_START)
        if markers[0] != EPOCH_START or len(imagery) == 0:
            raise ValueError(f"Subject {subject}, trial {trial_id}: epoch-start or imagery marker not found")
        # Stray markers occur late in a few epochs; the first one is the onset.
        trials.append(Trial(eeg[start:start + epoch:step], int(imagery[0]) // step, label, subject, 0, trial_id))
    return trials


def read_trials(data_dir, subject: int, config: DataConfig) -> list[Trial]:
    """The 40 trials of one subject, in recording order."""
    eeg, marker = read_recording(subject_file(data_dir, subject))
    return cut_trials(eeg, marker, read_labels(data_dir), subject, config)
