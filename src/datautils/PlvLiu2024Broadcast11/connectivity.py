"""Calculate core PLV connectivity and weighted graph adjacency for Liu2024.

This module trims Hilbert-transform boundaries, extracts instantaneous phase,
calculates a dense phase-locking-value matrix, and sparsifies it with a top-k
union rule. Inputs are individual EEG windows; file I/O and PyTorch Geometric
conversion are handled by other modules.

This is a self-contained copy of ``src.datautils.PlvLiu2024.core.connectivity``
kept so that ``PlvLiu2024Broadcast11`` has no import dependency on
``PlvLiu2024``. Keep in sync by hand if the original is ever fixed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def trim_signal_boundaries(signals: np.ndarray, trim_samples: int) -> np.ndarray:
    """Remove an equal number of time samples from both window boundaries."""

    values = np.asarray(signals)
    if values.ndim != 2:
        raise ValueError(f"Expected (channels, samples), got {values.shape}")
    if trim_samples < 0 or 2 * trim_samples >= values.shape[-1]:
        raise ValueError("Invalid boundary trim for signal length")
    return values[:, trim_samples:-trim_samples] if trim_samples else values


def calculate_instantaneous_phase(signals: np.ndarray) -> np.ndarray:
    """Calculate instantaneous phase along the final signal dimension."""

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite signals shaped (channels, samples)")
    analytic = np.asarray(hilbert(values, axis=-1), dtype=np.complex128)
    return np.arctan2(analytic.imag, analytic.real)


def calculate_plv_from_phase(phase: np.ndarray) -> np.ndarray:
    """Calculate a symmetric pairwise PLV matrix from instantaneous phases."""

    values = np.asarray(phase, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] == 0 or not np.isfinite(values).all():
        raise ValueError("Expected finite phase shaped (channels, samples)")
    unit_phase = np.exp(1j * values)
    plv = np.abs(unit_phase @ unit_phase.conj().T) / values.shape[-1]
    return np.clip(plv.real, 0.0, 1.0).astype(np.float32)


def calculate_plv(signals: np.ndarray) -> np.ndarray:
    """Calculate pairwise PLV directly from one multichannel EEG window."""

    return calculate_plv_from_phase(calculate_instantaneous_phase(signals))


def validate_plv_matrix(plv_matrix: np.ndarray, expected_nodes: int) -> None:
    """Verify PLV dimensions, range, symmetry, diagonal, and finite values."""

    plv = np.asarray(plv_matrix)
    if plv.shape != (expected_nodes, expected_nodes):
        raise ValueError(f"Unexpected PLV shape: {plv.shape}")
    if not np.isfinite(plv).all() or np.any(plv < 0) or np.any(plv > 1):
        raise ValueError("PLV contains invalid values")
    if not np.allclose(plv, plv.T, atol=1e-6):
        raise ValueError("PLV matrix is not symmetric")
    if not np.allclose(np.diag(plv), 1.0, atol=1e-6):
        raise ValueError("PLV diagonal is not one")


def create_threshold_adjacency(plv_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Create a symmetric BINARY adjacency by thresholding and binarizing PLV.

    This intentionally differs from PlvLiu2024's own create_threshold_adjacency,
    which keeps the continuous PLV weight for every retained edge. The
    manuscript's own graph-topology ablation explicitly describes
    "binarization strategies... to obtain graph adjacency matrices," and
    EEGGCN1 consumes a true 0/1 adjacency -- so every retained edge here
    becomes exactly 1.0 regardless of its PLV magnitude.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    plv = np.asarray(plv_matrix)
    adjacency = np.where(plv >= threshold, 1.0, 0.0).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def validate_adjacency_matrix(
    adjacency: np.ndarray,
    *,
    plv_matrix: np.ndarray,
    expected_nodes: int,
    require_no_isolated_nodes: bool = True,
    binary: bool = False,
) -> None:
    """Verify weighted or binary adjacency shape, symmetry, and coverage.

    A thresholded, binarized graph (see create_threshold_adjacency) can have
    isolated nodes when a window is weakly synchronized -- pass
    require_no_isolated_nodes=False for that case, since it is expected, not
    an error. Pass binary=True to check retained entries equal 1.0 instead of
    the original PLV magnitude.
    """

    matrix = np.asarray(adjacency)
    plv = np.asarray(plv_matrix)
    if matrix.shape != (expected_nodes, expected_nodes):
        raise ValueError(f"Unexpected adjacency shape: {matrix.shape}")
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("Adjacency contains invalid values")
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError("Adjacency matrix is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("Adjacency contains self-connections")
    retained = matrix > 0
    if binary:
        if not np.array_equal(matrix[retained], np.ones_like(matrix[retained])):
            raise ValueError("Binary adjacency must contain only 0/1 edge weights")
    elif not np.allclose(matrix[retained], plv[retained], atol=1e-6):
        raise ValueError("Adjacency edge weights do not match PLV")
    if require_no_isolated_nodes and np.any(np.count_nonzero(matrix, axis=1) == 0):
        raise ValueError("Adjacency contains isolated nodes")
