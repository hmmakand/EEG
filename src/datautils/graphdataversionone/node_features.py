"""Calculate the six inspected features from EEG or surface-Laplacian trials."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.signal import welch
from scipy.stats import entropy

from .config import BANDS, POWER_FLOOR, PSD_OVERLAP, PSD_WINDOW_SECONDS


def compute_psd(trial: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate each channel's power spectral density using the inspected Welch settings.

    Parameters
    ----------
    trial : ndarray, shape (n_channels, n_samples)
        One aligned motor-imagery window. EEG must be in V and CSD must be
        in V/m². The caller preserves the dataset channel order.
    sfreq : float
        Sampling frequency in Hz; the dataset uses 500 Hz.

    Returns
    -------
    frequencies : ndarray of float64, shape (n_frequencies,)
        Ascending one-sided frequency bins in Hz.
    psd : ndarray of float64, shape (n_channels, n_frequencies)
        Density in V²/Hz for EEG or (V/m²)²/Hz for CSD, in input channel order.

    Raises
    ------
    ValueError
        If the signal or sampling rate is invalid or produces nonfinite PSD.

    Notes
    -----
    Matches the notebook: Welch with a periodic Hann window, two-second
    segments (shortened only if necessary), 50% overlap, constant detrending,
    one-sided density scaling, and mean averaging. No CSD, filtering,
    reference change, normalization, disk writes, or input mutation occurs.
    """
    signal = np.asarray(trial, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] < 2:
        raise ValueError("trial must have shape (n_channels, n_samples) with at least two samples")
    if not np.isfinite(signal).all():
        raise ValueError("Node input contains nonfinite samples")
    if not np.isfinite(sfreq) or sfreq < 2 * BANDS[-1][1]:
        raise ValueError("Sampling rate must support the highest feature frequency")

    nperseg = min(round(PSD_WINDOW_SECONDS * sfreq), signal.shape[1])
    frequencies, psd = welch(
        signal,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=int(nperseg * PSD_OVERLAP),
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
        average="mean",
    )
    if not np.isfinite(psd).all():
        raise ValueError("Welch PSD contains nonfinite values")
    return frequencies, psd


def _validate_psd(psd: np.ndarray, frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return a nonnegative PSD and its matching frequency vector.

    Parameters
    ----------
    psd : ndarray, shape (n_channels, n_frequencies)
        Spectral density in squared SI signal units per Hz.
    frequencies : ndarray, shape (n_frequencies,)
        Strictly increasing nonnegative frequencies in Hz.

    Returns
    -------
    psd, frequencies : tuple of ndarray
        Float64 arrays with their original ordering and units.

    Raises
    ------
    ValueError
        If dimensions, values, or frequency ordering are invalid.

    Notes
    -----
    Inputs are not modified, and no files are written.
    """
    power = np.asarray(psd, dtype=np.float64)
    freqs = np.asarray(frequencies, dtype=np.float64)
    if power.ndim != 2 or power.shape[0] == 0 or freqs.ndim != 1 or power.shape[1] != freqs.size:
        raise ValueError("PSD must have shape (n_channels, n_frequencies) matching frequencies")
    if freqs.size < 2 or not np.isfinite(freqs).all() or np.any(freqs < 0) or np.any(np.diff(freqs) <= 0):
        raise ValueError("PSD frequencies must be finite, nonnegative, and strictly increasing")
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("PSD must contain finite nonnegative power")
    return power, freqs


def compute_band_powers(psd: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Integrate the five inspected frequency bands and convert to squared micro-units.

    Parameters
    ----------
    psd : ndarray, shape (n_channels, n_frequencies)
        EEG density in V²/Hz or CSD density in (V/m²)²/Hz.
    frequencies : ndarray, shape (n_frequencies,)
        Matching ascending frequency bins in Hz from compute_psd.

    Returns
    -------
    band_powers : ndarray of float64, shape (n_channels, 5)
        Linear delta, theta, alpha, beta, and gamma powers in µV² for EEG
        or (µV/m²)² for CSD. No logarithm or power floor is applied here.

    Raises
    ------
    ValueError
        If PSD data or frequency bins are invalid, a band has fewer than
        two bins, or integration produces invalid powers.

    Notes
    -----
    Simpson integration includes both endpoints of each configured band,
    matching the notebook. Multiplication by 1e12 converts squared SI
    units to squared micro-units. Inputs and files are not modified.
    """
    psd, frequencies = _validate_psd(psd, frequencies)
    band_powers = []
    for low_hz, high_hz in BANDS:
        mask = (frequencies >= low_hz) & (frequencies <= high_hz)
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"Insufficient Welch bins for band {low_hz}–{high_hz} Hz")
        power = simpson(psd[:, mask], x=frequencies[mask], axis=-1) * 1e12
        band_powers.append(power)
    powers = np.column_stack(band_powers)
    if not np.isfinite(powers).all() or np.any(powers < 0):
        raise ValueError("Integrated band powers must be finite and nonnegative")
    return powers


def compute_spectral_entropy(psd: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Calculate normalized Shannon entropy of each channel's 1–40 Hz spectrum.

    Parameters
    ----------
    psd : ndarray, shape (n_channels, n_frequencies)
        Nonnegative spectral density in consistent units for each channel.
    frequencies : ndarray, shape (n_frequencies,)
        Matching uniformly spaced frequency bins in Hz from compute_psd.

    Returns
    -------
    normalized_entropy : ndarray of float64, shape (n_channels,)
        Dimensionless entropy in [0, 1], retaining input channel order.

    Raises
    ------
    ValueError
        If PSD data are invalid, bins are not uniformly spaced, or the
        selected range has insufficient bins or zero total power.

    Notes
    -----
    Divide 1–40 Hz PSD samples by their sum, calculate base-two Shannon
    entropy, then divide by log2(number of selected bins). Band endpoints
    are included. No input mutation, logarithmic power conversion, or
    disk writes occur.
    """
    psd, frequencies = _validate_psd(psd, frequencies)
    if not np.allclose(np.diff(frequencies), frequencies[1] - frequencies[0], rtol=1e-10, atol=1e-12):
        raise ValueError("Spectral entropy requires uniformly spaced PSD frequencies")

    entropy_mask = (frequencies >= 1.0) & (frequencies <= 40.0)
    entropy_psd = psd[:, entropy_mask]
    total_power = entropy_psd.sum(axis=1, keepdims=True)
    if entropy_psd.shape[1] < 2 or np.any(total_power <= 0):
        raise ValueError("Cannot calculate spectral entropy from zero or insufficient PSD power")
    probabilities = entropy_psd / total_power
    normalized_entropy = entropy(probabilities, base=2, axis=1) / np.log2(entropy_psd.shape[1])
    if not np.isfinite(normalized_entropy).all():
        raise ValueError("Normalized spectral entropy contains nonfinite values")
    if np.any(normalized_entropy < -1e-12) or np.any(normalized_entropy > 1 + 1e-12):
        raise ValueError("Normalized spectral entropy is outside [0, 1]")
    return np.clip(normalized_entropy, 0.0, 1.0)


def build_node_features(trial: np.ndarray, sfreq: float) -> np.ndarray:
    """Combine five log-band powers and normalized spectral entropy per node.

    Parameters
    ----------
    trial : ndarray, shape (n_channels, n_samples)
        One aligned motor-imagery window, in V for EEG or V/m² for CSD.
        Channel order is preserved and input values are not modified.
    sfreq : float
        Sampling frequency in Hz; the dataset uses 500 Hz.

    Returns
    -------
    features : ndarray of float64, shape (n_channels, 6)
        Delta, theta, alpha, beta, and gamma powers in dB, followed by
        dimensionless spectral entropy. EEG powers use a reference of
        1 µV²; CSD powers use 1 (µV/m²)². Variant-specific reference units
        must be recorded in dataset metadata.

    Raises
    ------
    ValueError
        If the input signal, PSD, or resulting features are invalid.

    Notes
    -----
    Both calculations reuse one Welch PSD. Band powers in squared
    micro-units are floored at 1e-12 before applying 10*log10. CSD,
    filtering, reference changes, learned normalization, and disk writes
    are responsibilities of other pipeline stages.
    """
    frequencies, psd = compute_psd(trial, sfreq)
    band_powers = compute_band_powers(psd, frequencies)
    log_band_powers = 10 * np.log10(np.maximum(band_powers, POWER_FLOOR))
    normalized_entropy = compute_spectral_entropy(psd, frequencies)
    features = np.column_stack([log_band_powers, normalized_entropy])
    if not np.isfinite(features).all():
        raise ValueError("Node features contain nonfinite values")
    return features
