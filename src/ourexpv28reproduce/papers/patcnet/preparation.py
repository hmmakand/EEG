"""Trials as PA-TCNet's model receives them: [trials, channels, samples] and labels, per subject.

Paper, section 3.2: "the raw EEG signals were first band-pass filtered between
8 and 30 Hz and then downsampled to 250 Hz. All channels were subsequently
re-referenced by common average referencing and baseline corrected [...].
Finally, independent component analysis (ICA) was applied [...]."

The readers already return 250 Hz (the published signals hold nothing above
40 Hz, so the order of filtering and down-sampling does not matter). Filter
type and order, the baseline and the units are not given; the choices made
here are listed in README.md.
"""
import numpy as np
from scipy import signal

from ...datascripts import READERS, SAMPLE_RATE, liu2024
from .settings import DATASETS

BAND = (8.0, 30.0)
FILTER_ORDER = 4          # assumption: zero-phase Butterworth
MICROVOLTS = 1e6          # assumption: the model receives microvolts


def prepare_trials(trials, window_seconds):
    """[trials, channels, samples] float32 in microvolts, and the labels."""
    sos = signal.butter(FILTER_ORDER, BAND, "bandpass", fs=SAMPLE_RATE, output="sos")
    prepared = []
    for trial in trials:
        filtered = signal.sosfiltfilt(sos, trial.signal, axis=0)                # whole stretch, then crop
        if window_seconds is not None:
            start, stop = (trial.onset + int(round(s * SAMPLE_RATE)) for s in window_seconds)
            if start < 0 or stop > len(filtered):
                raise ValueError(f"Trial {trial.trial_id} of subject {trial.subject_id}: the window does not fit")
            filtered = filtered[start:stop]
        filtered = filtered - filtered.mean(axis=1, keepdims=True)              # common average reference
        filtered = filtered - filtered.mean(axis=0, keepdims=True)              # baseline: mean of the stretch
        prepared.append(filtered.T * MICROVOLTS)
    return np.stack(prepared).astype(np.float32), np.array([t.label for t in trials], dtype=np.int64)


def load_subject(data_dir, dataset: str, subject: int):
    """(x [trials, channels, samples], y) of one subject, prepared as in the paper."""
    settings = DATASETS[dataset]
    if dataset == "liu2024":
        trials = liu2024.read_trials(data_dir, subject, liu2024.CHANNELS_WITH_REFERENCE)   # the 30 channels
    else:
        trials = READERS[dataset].read_trials(data_dir, subject)
    x, y = prepare_trials(trials, settings.window_seconds)
    if x.shape[1:] != (settings.channels, settings.samples):
        raise ValueError(f"{dataset} subject {subject}: shape {x.shape[1:]}, the paper has "
                         f"{(settings.channels, settings.samples)}")
    return x, y
