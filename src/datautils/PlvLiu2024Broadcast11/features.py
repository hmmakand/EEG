"""Explicit rules for constructing the inferred 29-node by 11-feature matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manuscript_features import (
    MANUSCRIPT_NODE_FEATURE_NAMES,
    BetweennessConfig,
    PSDConfig,
    calculate_manuscript_candidate_features,
)
from .topology_features import TOPOLOGY_FEATURE_NAMES, TopologyFeatureConfig


LOCAL_FEATURE_NAMES = MANUSCRIPT_NODE_FEATURE_NAMES
BROADCAST_FEATURE_NAMES = TOPOLOGY_FEATURE_NAMES
NODE_FEATURE_NAMES = (*LOCAL_FEATURE_NAMES, *BROADCAST_FEATURE_NAMES)

PSD_CONFIGURATION = PSDConfig(
    reducer="mean_density",
    frequency_band_hz=(31.0, 40.0),
)
BETWEENNESS_CONFIGURATION = BetweennessConfig(
    graph_mode="threshold_binary", threshold=0.3
)
"""Must stay numerically in sync with Broadcast11Config.plv_threshold --
they are independent settings, not structurally linked."""
TOPOLOGY_CONFIGURATION = TopologyFeatureConfig()


@dataclass(frozen=True)
class Broadcast11FeatureResult:
    """Eleven node columns and the seven graph scalars used for broadcasting."""

    node_features: np.ndarray
    broadcast_source_features: np.ndarray
    node_feature_names: tuple[str, ...] = NODE_FEATURE_NAMES


def calculate_broadcast_11_features(
    signals: np.ndarray,
    plv_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    sampling_frequency_hz: float,
) -> Broadcast11FeatureResult:
    """Combine four local columns with seven repeated whole-graph descriptors."""

    manuscript = calculate_manuscript_candidate_features(
        signals,
        plv_matrix,
        adjacency_matrix,
        sampling_frequency_hz,
        psd_config=PSD_CONFIGURATION,
        betweenness_config=BETWEENNESS_CONFIGURATION,
        topology_config=TOPOLOGY_CONFIGURATION,
        std_ddof=0,
    )
    local = np.asarray(manuscript.node_features, dtype=np.float32)
    topology = np.asarray(manuscript.graph_features, dtype=np.float32)
    if local.ndim != 2 or local.shape[1] != len(LOCAL_FEATURE_NAMES):
        raise ValueError(f"Expected local features shaped (nodes, 4), got {local.shape}")
    if topology.shape != (len(BROADCAST_FEATURE_NAMES),):
        raise ValueError(f"Expected seven topology values, got {topology.shape}")
    broadcast = np.broadcast_to(topology, (local.shape[0], len(topology)))
    combined = np.concatenate((local, broadcast), axis=1).astype(
        np.float32, copy=False
    )
    if combined.shape != (local.shape[0], len(NODE_FEATURE_NAMES)):
        raise RuntimeError(f"Unexpected broadcast-11 shape: {combined.shape}")
    if not np.isfinite(combined).all():
        raise ValueError("Broadcast-11 features contain non-finite values")
    return Broadcast11FeatureResult(
        node_features=combined,
        broadcast_source_features=topology,
    )
