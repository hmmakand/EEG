"""Read a Liu et al. 2024 recording and cut an imagery and a rest sample from every trial.

Liu H. et al., "An EEG motor imagery dataset for brain computer interface in
acute stroke patients", Scientific Data 11, 131 (2024). Files as downloaded
by MOABB: one EDF per subject.

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

Which hand was imagined (left in even trials, right in odd ones) is not used.
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
EPOCH_START, IMAGERY_START = 1, 2    # marker values
MARKER_SCALE = 1e6                   # the reader returns the marker in volts; it was stored in microvolts
IMAGERY, REST = 1, 0                 # labels


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
    """Split a recording into its epochs and take two samples from each.

    The imagery sample starts `imagery_start` seconds after the epoch's imagery
    onset and the rest sample `rest_start` seconds after it, in the break. An
    epoch whose break does not fit (the imagery started late) gives no sample
    at all, so that the two classes stay balanced. `sample_id` is 2 * trial
    for the imagery and 2 * trial + 1 for the rest sample.

    The published signal has no content above 40 Hz, so taking every second
    sample (500 Hz -> 250 Hz) loses nothing and needs no further filter.
    """
    step, remainder = divmod(SOURCE_RATE, config.sample_rate)
    epoch = config.trial_samples * step
    if remainder or len(eeg) % epoch:
        raise ValueError(f"Subject {subject}: {len(eeg)} samples are not a whole number of {epoch}-sample epochs, "
                         "or the sample rate does not divide 500 Hz")
    rate = config.sample_rate
    imagery, rest = (int(round(seconds * rate)) for seconds in (config.imagery_start, config.rest_start))
    samples = []
    for trial_id in range(len(eeg) // epoch):
        begin = trial_id * epoch
        markers = marker[begin:begin + epoch]
        imagery_markers = np.flatnonzero(markers == IMAGERY_START)
        if markers[0] != EPOCH_START or len(imagery_markers) == 0:
            raise ValueError(f"Subject {subject}, trial {trial_id}: epoch-start or imagery marker not found")
        onset = int(imagery_markers[0]) // step     # stray markers occur late in a few epochs; the first is the onset
        if onset + rest + config.window_samples > config.trial_samples:
            continue
        signal = eeg[begin:begin + epoch:step]
        samples.append(Sample(signal, onset + imagery, IMAGERY, subject, trial_id, 2 * trial_id))
        samples.append(Sample(signal, onset + rest, REST, subject, trial_id, 2 * trial_id + 1))
    return samples


def read_samples(data_dir, subject: int, config: DataConfig) -> list[Sample]:
    """The imagery and rest samples of one subject, in recording order."""
    eeg, marker = read_recording(subject_file(data_dir, subject))
    return cut_samples(eeg, marker, subject, config)
