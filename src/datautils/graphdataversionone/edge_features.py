"""Calculate the notebook's five-band connectivity for a shared complete graph."""

from __future__ import annotations

import numpy as np
from mne_connectivity import spectral_connectivity_time

from .config import BANDS, CONNECTIVITY_SETTINGS, FREQUENCIES


def build_edge_index(n_channels: int = 29) -> np.ndarray:
    """Build a complete directed edge list with no self-loops.

    Parameters
    ----------
    n_channels : int, default 29
        Number of nodes, indexed in the same order as the signal channels.

    Returns
    -------
    edge_index : ndarray of int64, shape (2, n_channels * (n_channels - 1))
        Upper-triangle pairs in NumPy order followed by their reverses.
        For 29 channels, 406 unique pairs become 812 directed edges.

    Raises
    ------
    ValueError
        If fewer than two channels or a noninteger count is supplied.

    Notes
    -----
    This ordering matches the inspected notebook and is shared by every
    trial and connectivity variant. No file is written.
    """
    if isinstance(n_channels, bool) or not isinstance(n_channels, (int, np.integer)) or n_channels < 2:
        raise ValueError("n_channels must be an integer of at least two")
    source, target = np.triu_indices(n_channels, k=1)
    return np.vstack([np.r_[source, target], np.r_[target, source]]).astype(np.int64)


def compute_connectivity(trial: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """Calculate the three inspected connectivity methods for one signal window.

    Parameters
    ----------
    trial : ndarray, shape (n_channels, n_samples)
        One EEG window in V or a matching CSD window in V/m². Channel order
        and trial boundaries must match the node features. Values are not
        changed in place.
    sfreq : float
        Sampling frequency in Hz; the dataset uses 500 Hz.

    Returns
    -------
    connectivity : dict of str to ndarray of float64
        Keys ``wpli``, ``plv``, and ``icoh_abs`` each map to an array of shape
        (5, n_channels, n_channels), in the configured band order. Values
        are dimensionless, symmetric, in [0, 1], and zero on the diagonal.

    Raises
    ------
    ValueError
        If input samples or the connectivity output are invalid.

    Notes
    -----
    Uses spectral_connectivity_time on one epoch with multitaper mode,
    frequencies 1–40 Hz in 1-Hz steps, three cycles, time-bandwidth 2.0,
    time smoothing 0.5 s, and explicit within-band averaging. Both endpoints belong
    to each band, as in the notebook. All three methods share one spectral
    calculation. Explicit band averaging avoids the incorrect upper-bound
    slicing in mne-connectivity 0.9.0 _foi_average. Absolute iCoh is the magnitude AFTER signed iCoh has been
    averaged within each band; it is not mean absolute iCoh. The stored
    lower triangle is mirrored without estimating directionality.
    No extra filtering, CSD, graph thresholding, or disk writes occur here.
    """
    signal = np.asarray(trial, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] < 2 or signal.shape[1] < 2:
        raise ValueError("Connectivity input must contain at least two channels and samples")
    if not np.isfinite(signal).all():
        raise ValueError("Connectivity input contains nonfinite samples")
    if not np.isfinite(sfreq) or sfreq < 2 * BANDS[-1][1]:
        raise ValueError("Sampling rate must support the highest connectivity frequency")
    if signal.shape[1] / sfreq < CONNECTIVITY_SETTINGS["n_cycles"] / FREQUENCIES[0]:
        raise ValueError("Connectivity needs at least three seconds for three cycles at 1 Hz")
    if np.any(np.ptp(signal, axis=-1) == 0):
        raise ValueError("Connectivity is undefined for a constant channel")

    results = spectral_connectivity_time(
        data=signal[np.newaxis, :, :],
        freqs=np.asarray(FREQUENCIES, dtype=np.float64),
        method=["wpli", "plv", "imcoh"],
        sfreq=sfreq,
        fmin=tuple(band[0] for band in BANDS),
        fmax=tuple(band[1] for band in BANDS),
        sm_freqs=1,
        sm_kernel="hanning",
        padding=0.0,
        decim=1,
        n_jobs=1,
        verbose=False,
        **CONNECTIVITY_SETTINGS,
    )
    connectivity = {}
    n_channels = signal.shape[0]
    for name, result in zip(("wpli", "plv", "icoh_abs"), results, strict=True):
        triangle = result.get_data(output="dense")[0]
        if triangle.shape != (n_channels, n_channels, len(FREQUENCIES)):
            raise ValueError(f"Unexpected {name} connectivity shape: {triangle.shape}")
        # Average explicit inclusive bands. mne-connectivity 0.9.0
        # _foi_average doubles the upper index for multi-bin bands.
        freqs = np.asarray(FREQUENCIES)
        triangle = np.stack([triangle[..., (freqs >= low) & (freqs <= high)].mean(axis=-1)
                             for low, high in BANDS], axis=-1)
        if name == "icoh_abs":
            triangle = np.abs(triangle)
        adjacency = np.moveaxis(triangle + triangle.transpose(1, 0, 2), -1, 0)
        diagonal = np.arange(n_channels)
        adjacency[:, diagonal, diagonal] = 0.0
        if not np.isfinite(adjacency).all():
            raise ValueError(f"{name} connectivity contains nonfinite values")
        if adjacency.min() < -1e-12 or adjacency.max() > 1 + 1e-12:
            raise ValueError(f"{name} connectivity is outside [0, 1]")
        connectivity[name] = np.clip(adjacency, 0.0, 1.0)
    return connectivity


def build_edge_features(
    connectivity: dict[str, np.ndarray], edge_index: np.ndarray
) -> dict[str, np.ndarray]:
    """Gather five connectivity values for each edge in the shared ordering.

    Parameters
    ----------
    connectivity : dict of str to ndarray, shape (5, n_channels, n_channels)
        Symmetric, finite adjacency arrays returned by compute_connectivity.
        Each key identifies one dimensionless connectivity method.
    edge_index : ndarray of integer, shape (2, n_edges)
        Shared source/target node indices returned by build_edge_index.

    Returns
    -------
    features : dict of str to ndarray of float64, shape (n_edges, 5)
        Per-method edge attributes aligned column-by-column to edge_index.
        For 29 channels each array has shape (812, 5).

    Raises
    ------
    ValueError
        If indices, adjacency dimensions, symmetry, or values are invalid.

    Notes
    -----
    This function does not alter connectivity arrays or save files. The
    caller uses the same edge_index for both EEG and CSD variants.
    """
    indices = np.asarray(edge_index)
    if indices.ndim != 2 or indices.shape[0] != 2 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("edge_index must be an integer array with shape (2, n_edges)")
    if indices.shape[1] == 0 or np.any(indices < 0) or np.any(indices[0] == indices[1]):
        raise ValueError("edge_index must contain nonnegative node pairs without self-loops")
    features = {}
    for name, values in connectivity.items():
        adjacency = np.asarray(values, dtype=np.float64)
        if (
            adjacency.ndim != 3
            or adjacency.shape[0] != len(BANDS)
            or adjacency.shape[1] != adjacency.shape[2]
            or np.any(indices >= adjacency.shape[1])
        ):
            raise ValueError(f"Invalid {name} adjacency shape or out-of-range edge indices")
        if not np.isfinite(adjacency).all() or np.any(adjacency < 0) or np.any(adjacency > 1):
            raise ValueError(f"{name} adjacency must contain finite values in [0, 1]")
        if not np.allclose(adjacency, adjacency.transpose(0, 2, 1), rtol=0, atol=1e-12):
            raise ValueError(f"{name} adjacency must be symmetric")
        if np.any(np.diagonal(adjacency, axis1=1, axis2=2) != 0):
            raise ValueError(f"{name} adjacency must have zero diagonal")
        features[name] = np.ascontiguousarray(adjacency[:, indices[0], indices[1]].T)
    return features
