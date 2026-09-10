"""Core graph-level persistent-homology features inspired by Zhang et al. (2026).

These descriptors summarize a complete PLV graph. They are deliberately not
presented as node features because the manuscript does not define a mapping
from a persistence diagram back to individual EEG channels.

This is a self-contained copy of
``src.datautils.PlvLiu2024.core.topology_features`` kept so that
``PlvLiu2024Broadcast11`` has no import dependency on ``PlvLiu2024``. Keep in
sync by hand if the original is ever fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from ripser import ripser


TOPOLOGY_FEATURE_NAMES = (
    "PersistenceEntropy_0",
    "landscape1_0",
    "landscape1_1",
    "landscape2_0",
    "landscape2_1",
    "betti_0",
    "betti_1",
)


@dataclass(frozen=True)
class TopologyFeatureConfig:
    """Explicit assumptions for the manuscript-candidate PH calculation."""

    input_connectivity: Literal["full_plv"] = "full_plv"
    distance_definition: Literal["one_minus_absolute_connectivity"] = (
        "one_minus_absolute_connectivity"
    )
    homology_dimensions: tuple[int, int] = (0, 1)
    reduced_homology: bool = True
    maximum_filtration: float = 1.0
    coefficient: int = 2
    minimum_persistence: float = 0.0
    exclude_infinite_intervals: bool = True
    grid_minimum: float = 0.0
    grid_maximum: float = 1.0
    grid_points: int = 100
    amplitude_order: float = 1.0

    def __post_init__(self) -> None:
        if self.homology_dimensions != (0, 1):
            raise ValueError("This candidate implementation supports H0 and H1 only")
        if not self.reduced_homology:
            raise ValueError(
                "Non-reduced H0 requires an undocumented infinite-interval rule"
            )
        if self.maximum_filtration <= 0:
            raise ValueError("maximum_filtration must be positive")
        if self.coefficient < 2:
            raise ValueError("coefficient must be at least two")
        if self.minimum_persistence < 0:
            raise ValueError("minimum_persistence cannot be negative")
        if self.grid_minimum >= self.grid_maximum:
            raise ValueError("grid_minimum must be smaller than grid_maximum")
        if self.grid_points < 2:
            raise ValueError("grid_points must be at least two")
        if self.amplitude_order <= 0:
            raise ValueError("amplitude_order must be positive")
        if not self.exclude_infinite_intervals:
            raise ValueError(
                "Infinite intervals cannot be summarized without a documented rule"
            )


@dataclass(frozen=True)
class TopologyFeatureResult:
    """Seven graph-level scalars and the H0/H1 diagrams that produced them."""

    values: np.ndarray
    names: tuple[str, ...]
    diagrams: tuple[np.ndarray, np.ndarray]
    configuration: TopologyFeatureConfig


def plv_to_distance(plv_matrix: np.ndarray) -> np.ndarray:
    """Convert a symmetric PLV strength matrix to manuscript dissimilarities."""

    plv = np.asarray(plv_matrix, dtype=np.float64)
    if plv.ndim != 2 or plv.shape[0] != plv.shape[1]:
        raise ValueError(f"PLV matrix must be square, got {plv.shape}")
    if not np.isfinite(plv).all():
        raise ValueError("PLV matrix contains non-finite values")
    if np.any(plv < 0) or np.any(plv > 1):
        raise ValueError("PLV values must lie in [0, 1]")
    if not np.allclose(plv, plv.T, atol=1e-6):
        raise ValueError("PLV matrix must be symmetric")
    distance = 1.0 - np.abs(plv)
    np.fill_diagonal(distance, 0.0)
    return distance


def _finite_diagram(
    diagram: np.ndarray, config: TopologyFeatureConfig
) -> np.ndarray:
    values = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    values = values[np.isfinite(values).all(axis=1)]
    persistence = values[:, 1] - values[:, 0]
    return values[persistence > config.minimum_persistence]


def _persistence_entropy(diagram: np.ndarray) -> float:
    if len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    probabilities = lifetimes / lifetimes.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _landscape_amplitude(
    diagram: np.ndarray,
    grid: np.ndarray,
    *,
    layers: int,
    order: float,
) -> float:
    if len(diagram) == 0:
        return 0.0
    births = diagram[:, 0, None]
    deaths = diagram[:, 1, None]
    mountains = np.maximum(
        np.minimum(grid[None, :] - births, deaths - grid[None, :]), 0.0
    )
    ranked = np.sort(mountains, axis=0)[::-1]
    selected = ranked[:layers]
    return float(np.sum(selected**order) ** (1.0 / order))


def _betti_amplitude(
    diagram: np.ndarray, grid: np.ndarray, *, order: float
) -> float:
    if len(diagram) == 0:
        return 0.0
    alive = (
        (diagram[:, 0, None] <= grid[None, :])
        & (grid[None, :] <= diagram[:, 1, None])
    )
    curve = alive.sum(axis=0, dtype=np.float64)
    return float(np.sum(curve**order) ** (1.0 / order))


def calculate_topology_features(
    plv_matrix: np.ndarray,
    config: TopologyFeatureConfig | None = None,
) -> TopologyFeatureResult:
    """Return seven candidate graph descriptors from a full PLV matrix."""

    resolved = config or TopologyFeatureConfig()
    distance = plv_to_distance(plv_matrix)
    persistence = ripser(
        distance,
        distance_matrix=True,
        maxdim=1,
        thresh=resolved.maximum_filtration,
        coeff=resolved.coefficient,
    )["dgms"]
    h0 = _finite_diagram(persistence[0], resolved)
    h1 = _finite_diagram(persistence[1], resolved)
    grid = np.linspace(
        resolved.grid_minimum, resolved.grid_maximum, resolved.grid_points
    )
    values = np.asarray(
        (
            _persistence_entropy(h0),
            _landscape_amplitude(
                h0, grid, layers=1, order=resolved.amplitude_order
            ),
            _landscape_amplitude(
                h1, grid, layers=1, order=resolved.amplitude_order
            ),
            _landscape_amplitude(
                h0, grid, layers=2, order=resolved.amplitude_order
            ),
            _landscape_amplitude(
                h1, grid, layers=2, order=resolved.amplitude_order
            ),
            _betti_amplitude(h0, grid, order=resolved.amplitude_order),
            _betti_amplitude(h1, grid, order=resolved.amplitude_order),
        ),
        dtype=np.float32,
    )
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError("Unexpected persistent-homology feature result")
    return TopologyFeatureResult(
        values=values,
        names=TOPOLOGY_FEATURE_NAMES,
        diagrams=(h0, h1),
        configuration=resolved,
    )
