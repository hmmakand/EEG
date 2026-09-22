"""PLV connectivity; labels are handled by trials.py."""

from typing import cast

import numpy as np
from scipy.signal import hilbert


def extract_analytic_signal(signal):
    """Return Hilbert analytic signal."""
    return cast(np.ndarray, hilbert(signal, axis=0))


def extract_phase(signal):
    """Extract instantaneous phase using Hilbert transform."""
    analytic_signal = extract_analytic_signal(signal)
    return np.angle(analytic_signal)


def compute_plv_matrix(signal, diagonal=0.0):
    """Compute Phase Locking Value (PLV) connectivity matrix."""

    phases = extract_phase(signal)

    nodes = signal.shape[1]
    matrix = np.zeros((nodes, nodes))

    for source in range(nodes):
        for target in range(source + 1, nodes):

            phase_diff = phases[:, target] - phases[:, source]

            value = np.abs(
                np.mean(np.exp(1j * phase_diff))
            )

            matrix[source, target] = value
            matrix[target, source] = value

    np.fill_diagonal(matrix, diagonal)

    return matrix


def compute_connectivity(signal, config):
    """Compute PLV using the configured diagonal."""
    if config.method != "plv":
        raise ValueError("Connectivity method must be plv")
    return compute_plv_matrix(signal, config.diagonal)
