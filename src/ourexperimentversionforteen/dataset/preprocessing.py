"""Signal filtering with an explicit time axis."""
from dataclasses import replace
import numpy as np
from scipy import signal


def design_bandpass(edges, sample_rate, order=5):
    return signal.butter(order, edges, "bandpass", fs=sample_rate, output="sos")


def apply_bandpass(data, edges, sample_rate, order=5, time_axis=0):
    return signal.sosfiltfilt(design_bandpass(edges, sample_rate, order), data, axis=time_axis)


def preprocess_trial(trial, sample_rate, config):
    return replace(trial, signal=apply_bandpass(trial.signal, config.edges, sample_rate, config.order))


def pad_trials(trials):
    """Return [trials, samples, channels], zero padded to the longest trial."""
    if not trials:
        raise ValueError("Cannot pad an empty trial collection")
    output = np.zeros((len(trials), max(len(t.signal) for t in trials), trials[0].signal.shape[1]))
    for index, trial in enumerate(trials):
        output[index, :len(trial.signal)] = trial.signal
    return output
