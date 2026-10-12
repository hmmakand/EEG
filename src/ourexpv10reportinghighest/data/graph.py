"""One trial -> one graph of electrodes."""
import numpy as np
from scipy.integrate import simpson
from scipy.signal import hilbert, welch
import torch
from torch_geometric.data import Data

from .config import DataConfig
from .recordings import Trial
from .signal import analysed_signal, bandpass


def band_powers(window, config: DataConfig):
    """[samples, channels] -> [channels, bands]: Welch power integrated over each band."""
    length = min(int(config.welch_seconds * config.sample_rate), len(window))
    features = np.zeros((window.shape[1], len(config.feature_bands)))
    for band_index, (low, high) in enumerate(config.feature_bands):
        filtered = bandpass(window.T, (low, high), config.sample_rate, config.filter_order, time_axis=-1)
        for channel in range(window.shape[1]):
            frequencies, power = welch(filtered[channel], fs=config.sample_rate, window="hann", nperseg=length,
                                       noverlap=int(length * config.welch_overlap), detrend="constant")
            inside = (frequencies >= low) & (frequencies <= high)
            features[channel, band_index] = simpson(power[inside], dx=frequencies[1] - frequencies[0])
    return features


def phase_locking(window):
    """[samples, channels] -> [channels, channels] phase-locking value, zero diagonal."""
    phases = np.angle(hilbert(window, axis=0))
    channels = window.shape[1]
    matrix = np.zeros((channels, channels))
    for source in range(channels):
        for target in range(source + 1, channels):
            value = np.abs(np.mean(np.exp(1j * (phases[:, target] - phases[:, source]))))
            matrix[source, target] = matrix[target, source] = value
    return matrix


def plv_edges(window, threshold: float):
    """Ordered electrode pairs whose phase-locking value is above the threshold.

    The matrix is symmetric, so both directions of a pair are returned; there
    are no self-loops.
    """
    return np.array(np.nonzero(phase_locking(window) > threshold))


def build_graph(trial: Trial, aligned: Trial, config: DataConfig) -> Data:
    """`aligned` is the same trial after signal.align_subject.

    x          [22, 6]   log band powers of each electrode, from the aligned signal
    edge_index [2, E]    electrode pairs with PLV above the threshold, from the
                         signal of the physical electrodes (not aligned)

    The graph is taken from the unaligned signal because alignment decorrelates
    the channels: on aligned signals almost no pair passes the threshold.
    """
    features = np.log(np.clip(band_powers(analysed_signal(aligned.signal, config), config), 1e-10, None))
    edges = plv_edges(analysed_signal(trial.signal, config), config.plv_threshold)
    return Data(x=torch.tensor(features, dtype=torch.float32),
                edge_index=torch.tensor(edges, dtype=torch.long),
                y=torch.tensor(trial.label, dtype=torch.long),
                subject_id=trial.subject_id, session_id=trial.session_id, trial_id=trial.trial_id)
