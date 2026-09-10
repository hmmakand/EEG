"""Core candidate interpretations of manuscript-reported EEG features.

This is a self-contained copy of
``src.datautils.PlvLiu2024.core.manuscript_features`` kept so that
``PlvLiu2024Broadcast11`` has no import dependency on ``PlvLiu2024``. Keep in
sync by hand if the original is ever fixed. The only external dependency of
the original module -- ``calculate_mean_absolute_amplitude`` from
``PlvLiu2024.core.features`` -- is inlined below instead of copying that
entire module, since its other functions are unrelated to this feature set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
from scipy.signal import welch

from .topology_features import (
    TopologyFeatureConfig,
    calculate_topology_features,
)


MANUSCRIPT_NODE_FEATURE_NAMES = ("MAV", "STD", "PSD", "betweenness")


def _validated_signals(signals: np.ndarray) -> np.ndarray:
    """Return finite float64 signals after validating their dimensions."""

    validated = np.asarray(signals, dtype=np.float64)
    if validated.ndim != 2:
        raise ValueError(f"Expected (channels, samples), got {validated.shape}")
    if validated.shape[-1] < 3:
        raise ValueError("At least three samples are required")
    if not np.isfinite(validated).all():
        raise ValueError("Signals contain non-finite values")
    return validated


def calculate_mean_absolute_amplitude(signals: np.ndarray) -> np.ndarray:
    """Calculate mean absolute amplitude for every channel."""

    return np.mean(np.abs(_validated_signals(signals)), axis=-1)


@dataclass(frozen=True)
class PSDConfig:
    """An explicit rule for reducing each channel's PSD to one scalar."""

    reducer: Literal["mean_density", "band_power", "peak_density"]
    frequency_band_hz: tuple[float, float] | None = None
    window: str = "hann"
    nperseg: int | None = None
    noverlap: int | None = None
    nfft: int | None = None

    def __post_init__(self) -> None:
        if self.reducer == "band_power" and self.frequency_band_hz is None:
            raise ValueError("band_power requires frequency_band_hz")
        if self.frequency_band_hz is not None:
            low, high = self.frequency_band_hz
            if not 0 <= low < high:
                raise ValueError("frequency_band_hz must be increasing and nonnegative")
        if self.nperseg is not None and self.nperseg < 2:
            raise ValueError("nperseg must be at least two")
        if self.noverlap is not None and self.noverlap < 0:
            raise ValueError("noverlap cannot be negative")


@dataclass(frozen=True)
class BetweennessConfig:
    """An explicit graph and path-distance definition for betweenness."""

    graph_mode: Literal["top_k_binary", "threshold_binary", "full_weighted"]
    normalized: bool = True
    threshold: float | None = None
    distance_rule: Literal["one_minus_plv", "inverse_plv"] | None = None
    distance_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.graph_mode == "threshold_binary":
            if self.threshold is None or not 0 <= self.threshold <= 1:
                raise ValueError("threshold_binary requires a threshold in [0, 1]")
        elif self.threshold is not None:
            raise ValueError("threshold is only valid for threshold_binary")
        if self.graph_mode == "full_weighted" and self.distance_rule is None:
            raise ValueError("full_weighted requires an explicit distance_rule")
        if self.graph_mode != "full_weighted" and self.distance_rule is not None:
            raise ValueError("distance_rule is only valid for full_weighted")
        if self.distance_epsilon <= 0:
            raise ValueError("distance_epsilon must be positive")


@dataclass(frozen=True)
class PSDResult:
    frequencies: np.ndarray
    density: np.ndarray


@dataclass(frozen=True)
class ManuscriptFeatureResult:
    """Four node features and seven separate whole-graph features."""

    node_features: np.ndarray
    graph_features: np.ndarray
    node_feature_names: tuple[str, ...]
    graph_feature_names: tuple[str, ...]
    configuration: dict[str, object]


def calculate_mav(signals: np.ndarray) -> np.ndarray:
    """Return manuscript mean absolute value independently per channel."""

    return calculate_mean_absolute_amplitude(_validated_signals(signals))


def calculate_std(signals: np.ndarray, *, ddof: int = 0) -> np.ndarray:
    """Return channel standard deviations with an explicit delta degrees of freedom."""

    values = _validated_signals(signals)
    if not 0 <= ddof < values.shape[1]:
        raise ValueError("ddof must be smaller than the number of samples")
    return np.std(values, axis=1, ddof=ddof)


def calculate_psd_spectrum(
    signals: np.ndarray, sampling_frequency_hz: float, config: PSDConfig
) -> PSDResult:
    """Calculate Welch PSD without silently reducing its frequency dimension."""

    values = _validated_signals(signals)
    if sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be positive")
    frequencies, density = welch(
        values,
        fs=sampling_frequency_hz,
        window=config.window,
        nperseg=config.nperseg,
        noverlap=config.noverlap,
        nfft=config.nfft,
        axis=1,
        scaling="density",
    )
    return PSDResult(frequencies=frequencies, density=density)


def reduce_psd(result: PSDResult, config: PSDConfig) -> np.ndarray:
    """Apply the explicitly selected scalar PSD reducer to every channel."""

    frequencies = result.frequencies
    density = result.density
    selected = np.ones_like(frequencies, dtype=bool)
    if config.frequency_band_hz is not None:
        low, high = config.frequency_band_hz
        selected = (frequencies >= low) & (frequencies <= high)
    if not np.any(selected):
        raise ValueError("The requested frequency band contains no PSD bins")
    selected_density = density[:, selected]
    if config.reducer == "mean_density":
        return selected_density.mean(axis=1)
    if config.reducer == "peak_density":
        return selected_density.max(axis=1)
    return np.trapezoid(selected_density, frequencies[selected], axis=1)


def calculate_betweenness(
    plv_matrix: np.ndarray,
    adjacency_matrix: np.ndarray | None,
    config: BetweennessConfig,
) -> np.ndarray:
    """Return one betweenness value per channel using an explicit graph rule."""

    plv = np.asarray(plv_matrix, dtype=np.float64)
    if plv.ndim != 2 or plv.shape[0] != plv.shape[1]:
        raise ValueError("plv_matrix must be square")
    if not np.isfinite(plv).all() or not np.allclose(plv, plv.T, atol=1e-6):
        raise ValueError("plv_matrix must be finite and symmetric")
    if np.any(plv < 0) or np.any(plv > 1):
        raise ValueError("PLV values must lie in [0, 1]")

    if config.graph_mode == "top_k_binary":
        if adjacency_matrix is None:
            raise ValueError("top_k_binary requires adjacency_matrix")
        adjacency = np.asarray(adjacency_matrix)
        if adjacency.shape != plv.shape:
            raise ValueError("adjacency_matrix and plv_matrix shapes must match")
        graph = nx.from_numpy_array((adjacency > 0).astype(np.float64))
        weight = None
    elif config.graph_mode == "threshold_binary":
        assert config.threshold is not None
        adjacency = (plv >= config.threshold).astype(np.float64)
        np.fill_diagonal(adjacency, 0)
        graph = nx.from_numpy_array(adjacency)
        weight = None
    else:
        if config.distance_rule == "one_minus_plv":
            distance = np.maximum(1.0 - plv, config.distance_epsilon)
        else:
            distance = 1.0 / np.maximum(plv, config.distance_epsilon)
        np.fill_diagonal(distance, 0)
        graph = nx.from_numpy_array(distance)
        weight = "weight"

    centrality = nx.betweenness_centrality(
        graph, normalized=config.normalized, weight=weight
    )
    return np.asarray([centrality[index] for index in range(plv.shape[0])])


def calculate_manuscript_candidate_features(
    signals: np.ndarray,
    plv_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    sampling_frequency_hz: float,
    *,
    psd_config: PSDConfig,
    betweenness_config: BetweennessConfig,
    topology_config: TopologyFeatureConfig,
    std_ddof: int = 0,
) -> ManuscriptFeatureResult:
    """Calculate four node features and seven graph features without broadcasting."""

    spectrum = calculate_psd_spectrum(signals, sampling_frequency_hz, psd_config)
    topology = calculate_topology_features(plv_matrix, topology_config)
    node_features = np.column_stack(
        (
            calculate_mav(signals),
            calculate_std(signals, ddof=std_ddof),
            reduce_psd(spectrum, psd_config),
            calculate_betweenness(
                plv_matrix, adjacency_matrix, betweenness_config
            ),
        )
    ).astype(np.float32)
    if not np.isfinite(node_features).all():
        raise ValueError("Calculated manuscript candidate features are non-finite")
    return ManuscriptFeatureResult(
        node_features=node_features,
        graph_features=topology.values,
        node_feature_names=MANUSCRIPT_NODE_FEATURE_NAMES,
        graph_feature_names=topology.names,
        configuration={
            "psd": psd_config,
            "betweenness": betweenness_config,
            "topology": topology_config,
            "std_ddof": std_ddof,
            "global_features_broadcast_to_nodes": False,
        },
    )
