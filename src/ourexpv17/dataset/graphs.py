"""Edge selection and PyG conversion, including original edge behavior."""
import numpy as np
import torch
from torch_geometric.data import Data


def make_edge_mask(matrix, threshold, comparison="gt", self_loops=False):
    if comparison == "gt":
        mask = matrix > threshold
    elif comparison == "ge":
        mask = matrix >= threshold
    else:
        raise ValueError(f"Unknown threshold comparison: {comparison}")
    if not self_loops:
        np.fill_diagonal(mask, False)
    return mask


def connectivity_to_edges(matrix, config):
    if config.mode == "legacy":
        # First reproduce the NetworkX adjacency, then the trainer's >= pass.
        adjacency = np.where(make_edge_mask(matrix, config.threshold), matrix, 0)
        pairs = np.indices(matrix.shape).reshape(2, -1)
        keep = adjacency.ravel() >= config.threshold
        pairs[:, ~keep] = 0
        weights = np.where(keep, adjacency.ravel(), 0)
    elif config.mode == "threshold":
        mask = make_edge_mask(matrix, config.threshold, config.comparison, config.self_loops)
        if not config.directed:
            mask = mask | mask.T  # Both orientations for PyG message passing.
        pairs = np.array(np.nonzero(mask))
        weights = matrix[tuple(pairs)]
    else:
        raise ValueError(f"Unknown graph mode: {config.mode}")
    return torch.tensor(pairs, dtype=torch.long), torch.tensor(weights, dtype=torch.float32)


def make_pyg_graph(features, connectivity, trial, config):
    edges, weights = connectivity_to_edges(connectivity, config)
    graph = Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edges,
                 y=torch.tensor(trial.label, dtype=torch.long),
                 subject_id=trial.subject_id, session_id=trial.session_id, trial_id=trial.trial_id)
    if config.include_edge_weights:
        graph.edge_weight = weights
    return graph
