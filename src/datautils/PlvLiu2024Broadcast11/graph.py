"""Build validated core PLV graph components and PyG samples.

One source EEG window becomes node features, optional graph features, a dense
PLV matrix, and a PLV-threshold binary adjacency matrix. Sparse bidirectional
edges are exposed through a PyG ``Data`` object with provenance identifiers.

This is a self-contained copy of the parts of
``src.datautils.PlvLiu2024.core.graph`` that this package actually uses, kept
so that ``PlvLiu2024Broadcast11`` has no import dependency on ``PlvLiu2024``.
Keep in sync by hand if the original is ever fixed. ``create_graph_components``
(the unrelated six-feature ``current_v1`` path) is intentionally not copied.
``create_connectivity_components`` takes ``Broadcast11Config`` directly instead
of the generic ``PlvGraphConfig``, and ``validate_pyg_graph`` takes
``expected_nodes`` directly instead of a config object, since that was the
only field of the config it ever read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch_geometric.data import Data

from .connectivity import (
    calculate_plv,
    create_threshold_adjacency,
    trim_signal_boundaries,
    validate_adjacency_matrix,
    validate_plv_matrix,
)

if TYPE_CHECKING:
    from .config import Broadcast11Config


@dataclass(frozen=True)
class GraphComponents:
    """Dense, auditable representation of one processed EEG graph."""

    node_features: np.ndarray
    plv_matrix: np.ndarray
    adjacency_matrix: np.ndarray
    graph_features: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )


@dataclass(frozen=True)
class ConnectivityComponents:
    """PLV and sparse adjacency calculated once for one EEG trial."""

    plv_matrix: np.ndarray
    adjacency_matrix: np.ndarray


def create_connectivity_components(
    signals: np.ndarray, config: "Broadcast11Config"
) -> ConnectivityComponents:
    """Validate EEG and calculate reusable PLV and adjacency components."""

    values = np.asarray(signals)
    expected_shape = (config.expected_channels, config.expected_samples)
    if values.shape != expected_shape:
        raise ValueError(f"Expected EEG window shape {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("EEG window contains non-finite values")
    phase_signals = trim_signal_boundaries(values, config.hilbert_edge_trim_samples)
    plv = calculate_plv(phase_signals)
    adjacency = create_threshold_adjacency(plv, config.plv_threshold)
    validate_plv_matrix(plv, config.expected_channels)
    validate_adjacency_matrix(
        adjacency,
        plv_matrix=plv,
        expected_nodes=config.expected_channels,
        require_no_isolated_nodes=False,
        binary=True,
    )
    return ConnectivityComponents(plv, adjacency)


def adjacency_to_edges(adjacency: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense weighted adjacency to bidirectional PyG edge tensors."""

    matrix = np.asarray(adjacency)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Adjacency must be square, got {matrix.shape}")
    source, target = np.nonzero(matrix)
    edge_index = torch.as_tensor(np.vstack((source, target)), dtype=torch.long)
    edge_attr = torch.as_tensor(matrix[source, target], dtype=torch.float32).unsqueeze(-1)
    return edge_index, edge_attr


def validate_sparse_edges(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, adjacency: np.ndarray
) -> None:
    """Verify sparse edges exactly reconstruct a symmetric dense adjacency."""

    matrix = np.asarray(adjacency)
    expected_edges = int(np.count_nonzero(matrix))
    if edge_index.shape != (2, expected_edges):
        raise ValueError(f"Unexpected edge_index shape: {tuple(edge_index.shape)}")
    if edge_attr.shape != (expected_edges, 1):
        raise ValueError(f"Unexpected edge_attr shape: {tuple(edge_attr.shape)}")
    if edge_index.dtype != torch.long or edge_attr.dtype != torch.float32:
        raise ValueError("Unexpected sparse edge dtype")
    if torch.any(edge_index < 0) or torch.any(edge_index >= matrix.shape[0]):
        raise ValueError("Sparse edge contains an invalid node index")
    if torch.any(edge_index[0] == edge_index[1]):
        raise ValueError("Sparse edge representation contains self-edges")
    reconstructed = np.zeros_like(matrix, dtype=np.float32)
    reconstructed[
        edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    ] = edge_attr[:, 0].cpu().numpy()
    if not np.allclose(reconstructed, matrix, atol=1e-6):
        raise ValueError("Sparse edges do not reconstruct adjacency")


def create_pyg_graph(
    components: GraphComponents,
    *,
    label: int,
    subject_id: int,
    trial_index: int,
    window_index: int,
) -> Data:
    """Create one PyG graph with signal, connectivity, and provenance tensors."""

    edge_index, edge_attr = adjacency_to_edges(components.adjacency_matrix)
    validate_sparse_edges(edge_index, edge_attr, components.adjacency_matrix)
    attributes = dict(
        # ``torch.tensor`` intentionally copies memory-mapped read-only rows so
        # the resulting PyG tensor is safe to move, batch, or modify.
        x=torch.tensor(components.node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        subject_id=torch.tensor([subject_id], dtype=torch.long),
        trial_index=torch.tensor([trial_index], dtype=torch.long),
        window_index=torch.tensor([window_index], dtype=torch.long),
    )
    graph_features = np.asarray(components.graph_features)
    if graph_features.size:
        if graph_features.ndim != 1 or not np.isfinite(graph_features).all():
            raise ValueError("graph_features must be a finite vector")
        attributes["graph_features"] = torch.tensor(
            graph_features[None, :], dtype=torch.float32
        )
    return Data(**attributes)


def validate_pyg_graph(
    graph: Data,
    *,
    expected_nodes: int,
    expected_node_features: int,
    expected_graph_features: int = 0,
) -> None:
    """Verify required tensor fields and values of one generated PyG graph."""

    x = cast(torch.Tensor, graph.x)
    edge_index = cast(torch.Tensor, graph.edge_index)
    edge_attr = cast(torch.Tensor, graph.edge_attr)
    y = cast(torch.Tensor, graph.y)
    subject = cast(torch.Tensor, graph.subject_id)
    trial = cast(torch.Tensor, graph.trial_index)
    window = cast(torch.Tensor, graph.window_index)
    if any(value is None for value in (x, edge_index, edge_attr, y, subject, trial, window)):
        raise ValueError("PyG graph is missing a required tensor")
    if x.shape != (expected_nodes, expected_node_features):
        raise ValueError(f"Unexpected PyG node-feature shape: {tuple(x.shape)}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Unexpected PyG edge-index shape: {tuple(edge_index.shape)}")
    if edge_attr.shape != (edge_index.shape[1], 1):
        raise ValueError(f"Unexpected PyG edge-attribute shape: {tuple(edge_attr.shape)}")
    if not torch.isfinite(x).all() or not torch.isfinite(edge_attr).all():
        raise ValueError("PyG graph contains non-finite values")
    if torch.any(edge_attr < 0) or torch.any(edge_attr > 1):
        raise ValueError("PyG graph contains an invalid PLV edge weight")
    if torch.any(edge_index[0] == edge_index[1]):
        raise ValueError("PyG graph contains self-edges")
    if y.numel() != 1 or int(y.item()) not in (0, 1):
        raise ValueError("PyG graph has an invalid label")
    if subject.numel() != 1 or not 1 <= int(subject.item()) <= 50:
        raise ValueError("PyG graph has an invalid subject ID")
    if trial.numel() != 1 or not 0 <= int(trial.item()) < 40:
        raise ValueError("PyG graph has an invalid trial index")
    if window.numel() != 1 or int(window.item()) < 0:
        raise ValueError("PyG graph has an invalid source-window index")
    graph_features = getattr(graph, "graph_features", None)
    if expected_graph_features:
        if not isinstance(graph_features, torch.Tensor):
            raise ValueError("PyG graph is missing graph_features")
        if graph_features.shape != (1, expected_graph_features):
            raise ValueError(
                f"Unexpected graph-feature shape: {tuple(graph_features.shape)}"
            )
        if not torch.isfinite(graph_features).all():
            raise ValueError("PyG graph contains non-finite graph features")
    elif graph_features is not None:
        raise ValueError("PyG graph unexpectedly contains graph_features")
