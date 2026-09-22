"""Connectivity estimators; labels are handled by trials.py."""

from typing import cast

import numpy as np
from scipy.signal import hilbert, coherence
from .validation import validate_coherence_settings


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

def compute_ciplv_matrix(signal, diagonal=0.0, eps=1e-12):
    """Compute corrected imaginary Phase Locking Value (ciPLV)."""

    phases = extract_phase(signal)

    nodes = signal.shape[1]
    matrix = np.zeros((nodes, nodes))

    for source in range(nodes):
        for target in range(source + 1, nodes):

            phase_diff = (
                phases[:, target]
                - phases[:, source]
            )

            # Complex-valued PLV before taking magnitude
            complex_plv = np.mean(
                np.exp(1j * phase_diff)
            )

            real_part = np.real(complex_plv)
            imag_part = np.imag(complex_plv)

            denominator = np.sqrt(
                max(
                    1.0 - real_part**2,
                    eps
                )
            )

            value = (
                np.abs(imag_part)
                / denominator
            )

            matrix[source, target] = value
            matrix[target, source] = value

    np.fill_diagonal(matrix, diagonal)

    return matrix

def compute_pli_matrix(signal, diagonal=0.0):
    """Compute Phase Lag Index (PLI) connectivity matrix."""

    phases = extract_phase(signal)

    nodes = signal.shape[1]
    matrix = np.zeros((nodes, nodes))

    for source in range(nodes):
        for target in range(source + 1, nodes):

            phase_diff = phases[:, target] - phases[:, source]

            value = np.abs(
                np.mean(
                    np.sign(
                        np.sin(phase_diff)
                    )
                )
            )

            matrix[source, target] = value
            matrix[target, source] = value

    np.fill_diagonal(matrix, diagonal)

    return matrix


def compute_wpli_matrix(signal, diagonal=0.0, eps=1e-12):
    """Compute weighted Phase Lag Index (wPLI) connectivity matrix."""

    analytic_signal = extract_analytic_signal(signal)

    nodes = signal.shape[1]
    matrix = np.zeros((nodes, nodes))

    for source in range(nodes):
        for target in range(source + 1, nodes):

            # Complex cross-product
            cross_product = (
                analytic_signal[:, target]
                * np.conj(analytic_signal[:, source])
            )

            # Imaginary component
            imaginary = np.imag(cross_product)

            numerator = np.abs(np.sum(imaginary))
            denominator = np.sum(np.abs(imaginary))

            value = numerator / (denominator + eps)

            matrix[source, target] = value
            matrix[target, source] = value

    np.fill_diagonal(matrix, diagonal)

    return matrix


def compute_coherence_matrix(signal, sfreq, fmin, fmax, diagonal=0.0,
                             nperseg=256, noverlap=None, window="hann", detrend="constant"):
    """Band-average magnitude-squared coherence for [samples, channels] EEG.

    Invalid/zero-power estimates raise an error instead of silently losing edges.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2 or min(signal.shape) == 0 or not np.isfinite(signal).all():
        raise ValueError("Coherence requires finite, nonempty [samples, channels] signals")
    if not 0 <= diagonal <= 1:
        raise ValueError("Connectivity diagonal must be in [0, 1]")
    validate_coherence_settings(sfreq, fmin, fmax, nperseg, noverlap, window, detrend, len(signal))
    if np.any(np.ptp(signal, axis=0) == 0):
        raise ValueError("Coherence is undefined for constant or zero-power channels")
    nodes = signal.shape[1]
    matrix = np.zeros((nodes, nodes))
    for source in range(nodes):
        for target in range(source + 1, nodes):
            with np.errstate(divide="ignore", invalid="ignore"):
                freqs, values = coherence(signal[:, source], signal[:, target], fs=sfreq,
                                          nperseg=nperseg, noverlap=noverlap,
                                          window=window, detrend=detrend)
            band_values = values[(freqs >= fmin) & (freqs <= fmax)]
            if band_values.size == 0 or not np.isfinite(band_values).all():
                raise ValueError(f"Undefined coherence in selected band for channels {source}, {target}")
            value = float(np.clip(np.mean(band_values), 0.0, 1.0))
            matrix[source, target] = matrix[target, source] = value
    np.fill_diagonal(matrix, diagonal)
    return matrix


def compute_consensus_matrix(signal, diagonal=0.0):
    """Consensus connectivity: the elementwise minimum of PLV and wPLI.

    PLV is inflated by volume conduction - one source reaching two electrodes
    at zero lag scores near 1.0 with no interaction present. wPLI discards
    zero-lag coupling and rejects that artefact. Taking the minimum is an AND:
    a pair only counts as connected when both estimators agree.

    Converted to a distance with the usual 1 - |c|, this equals the elementwise
    maximum of the two individual distances, so volume-conducted pairs enter a
    Vietoris-Rips filtration late instead of immediately.

    Requires NARROWBAND input; band-pass before calling. Does not correct
    short-window bias.
    """

    plv_matrix = compute_plv_matrix(signal, diagonal=1.0)
    wpli_matrix = compute_wpli_matrix(signal, diagonal=1.0)

    matrix = np.minimum(np.abs(plv_matrix), np.abs(wpli_matrix))

    matrix = (matrix + matrix.T) / 2.0
    np.clip(matrix, 0.0, 1.0, out=matrix)
    np.fill_diagonal(matrix, diagonal)

    return matrix

def compute_reliability_gated_plv(signal, distance_matrix, diagonal=0.0,
                                  lambda_=0.5, sigma=0.05):
    """Spatially adaptive reliability-gated PLV.

    PLV remains the main connectivity measure. ciPLV and wPLI provide evidence
    about whether a connection is supported by non-zero-lag synchronization.
    Nearby electrode pairs receive stronger correction than distant ones.
    """

    plv = compute_plv_matrix(signal, diagonal=1.0)
    ciplv = compute_ciplv_matrix(signal, diagonal=1.0)
    wpli = compute_wpli_matrix(signal, diagonal=1.0)

    reliability = (ciplv + wpli) / 2.0
    leakage_risk = np.exp(-(distance_matrix ** 2) / (2.0 * sigma ** 2))

    matrix = plv * (1.0 - lambda_ * leakage_risk * (1.0 - reliability))
    matrix = (matrix + matrix.T) / 2.0

    np.clip(matrix, 0.0, 1.0, out=matrix)
    np.fill_diagonal(matrix, diagonal)

    return matrix


def compute_connectivity(signal, config, sample_rate=None):
    """Select connectivity estimator according to configuration."""

    if config.method == "plv":
        return compute_plv_matrix(
            signal,
            config.diagonal
        )

    if config.method == "ciplv":
        return compute_ciplv_matrix(
            signal,
            config.diagonal
        )

    if config.method == "pli":
        return compute_pli_matrix(
            signal,
            config.diagonal
        )

    if config.method == "wpli":
        return compute_wpli_matrix(
            signal,
            config.diagonal
        )

    if config.method == "coherence":
        if sample_rate is None:
            raise ValueError("Coherence requires the dataset sampling rate")
        return compute_coherence_matrix(
            signal, sample_rate, config.fmin, config.fmax, config.diagonal,
            config.nperseg, config.noverlap, config.window, config.detrend,
        )

    if config.method == "consensus":
        return compute_consensus_matrix(
            signal,
            config.diagonal
        )

    if config.method == "rplv":
        return compute_reliability_gated_plv(
            signal,
            config.diagonal
        )

    raise ValueError(
        f"Unknown connectivity method: {config.method}"
    )
