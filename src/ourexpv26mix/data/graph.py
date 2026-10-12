"""One aligned trial -> one graph of electrodes."""
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


def covariance(window):
    """[samples, channels] -> [channels, channels]; no mean removal, the signal is band-passed."""
    return window.T @ window / len(window)


EIGENVALUE_FLOOR = 1e-6    # relative to the largest eigenvalue; keeps the logarithm finite for a rank-deficient trial


def covariance_logarithm(matrix):
    """Matrix logarithm of a covariance matrix, [channels, channels].

    After alignment the subject's mean covariance is the identity, so this is
    the trial's covariance in the tangent space at the subject's mean
    (Barachant et al., 2012): a symmetric matrix that can be treated with
    ordinary linear operations.
    """
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, values.max() * EIGENVALUE_FLOOR)
    return (vectors * np.log(values)) @ vectors.T


def build_graph(trial: Trial, config: DataConfig) -> Data:
    """`trial` must already be aligned (see signal.align_subject).

    With C electrodes and W analysed samples (29 and 1000 for liu2024, 22 and 500 for bci2a):

    x          [C, 6]          band powers of each electrode
    edge_index [2, C * (C-1)]  every ordered electrode pair, no self-loops
    edge_attr  [C * (C-1), 2]  phase-locking value and covariance of the pair
    With config.tangent_features, x has a seventh column and edge_attr a third: the diagonal and the
    off-diagonal entries of the logarithm of the trial's covariance (see covariance_logarithm).
    signal     [C, W]          the analysed signal of each electrode
    """
    window = analysed_signal(trial, config)
    pair_matrices = [phase_locking(window), covariance(window)]
    node_features = band_powers(window, config)
    if config.tangent_features:
        tangent = covariance_logarithm(pair_matrices[1])
        pair_matrices.append(tangent)
        # The model takes the logarithm of the node features, so the diagonal is stored as its exponential.
        node_features = np.concatenate([node_features, np.exp(np.diag(tangent))[:, None]], axis=1)
    pair_features = np.stack(pair_matrices, axis=-1)
    pairs = np.array(np.nonzero(~np.eye(window.shape[1], dtype=bool)))
    return Data(x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(pairs, dtype=torch.long),
                edge_attr=torch.tensor(pair_features[tuple(pairs)], dtype=torch.float32),
                signal=torch.tensor(np.ascontiguousarray(window.T), dtype=torch.float32),
                y=torch.tensor(trial.label, dtype=torch.long),
                subject_id=trial.subject_id, session_id=trial.session_id, trial_id=trial.trial_id)
