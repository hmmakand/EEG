"""One aligned sample -> one graph of electrodes."""
import numpy as np
from scipy.integrate import simpson
from scipy.signal import hilbert, welch
import torch
from torch_geometric.data import Data

from .config import DataConfig
from .recording import Sample
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


def covariance(window):
    """[samples, channels] -> [channels, channels]; no mean removal, the signal is band-passed."""
    return window.T @ window / len(window)


def build_graph(sample: Sample, config: DataConfig) -> Data:
    """`sample` must already be aligned (see signal.align_subject).

    With C electrodes and W analysed samples (29 and 475 for liu2024, 22 and 500 for bci2a):

    x          [C, 6]          band powers of each electrode
    edge_index [2, C * (C-1)]  every ordered electrode pair, no self-loops
    edge_attr  [C * (C-1), 2]  phase-locking value and covariance of the pair
    signal     [C, W]          the analysed signal of each electrode
    y                          the label; see DataConfig.class_names
    """
    window = analysed_signal(sample, config)
    pair_features = np.stack([phase_locking(window), covariance(window)], axis=-1)
    pairs = np.array(np.nonzero(~np.eye(window.shape[1], dtype=bool)))
    return Data(x=torch.tensor(band_powers(window, config), dtype=torch.float32),
                edge_index=torch.tensor(pairs, dtype=torch.long),
                edge_attr=torch.tensor(pair_features[tuple(pairs)], dtype=torch.float32),
                signal=torch.tensor(np.ascontiguousarray(window.T), dtype=torch.float32),
                y=torch.tensor(sample.label, dtype=torch.long),
                subject_id=sample.subject_id, trial_id=sample.trial_id, sample_id=sample.sample_id)
