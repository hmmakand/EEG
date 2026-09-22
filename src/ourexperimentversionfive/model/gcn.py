"""EEGGCN1: sparse, edge-weighted graph convolutional classifier.

Architecture in the spirit of Zhang et al. (2026), Table 2: one weighted
``GCNConv`` layer, followed by BatchNorm/LeakyReLU/dropout, a flatten of all
node embeddings into one vector, and a two-layer fully connected classifier
ending in log-softmax. Unlike a dense ``DenseGCNConv`` model, this takes a
sparse ``x: (total_nodes, F)`` / ``edge_index: (2, total_edges)`` /
``edge_weight: (total_edges,)`` graph (or a batch of them, concatenated by
``torch_geometric``'s ``DataLoader``/``Batch``) -- there is no dense
adjacency matrix involved. Every graph must contain exactly ``nodes`` nodes,
so node embeddings can be reshaped back to ``(graphs, nodes, width)`` with
``to_dense_batch`` before the fully connected head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


EEGGCN1_NODES = 29
EEGGCN1_INPUT_FEATURES = 8
"""Node count and feature width of the without-CSD Graph-Liu2024-VersionTwo combination."""


@dataclass(frozen=True)
class EEGGCN1Config:
    """Architecture options for the sparse EEGGCN1 classifier.

    Defaults follow the standard GCN supervised-classification convention
    (Kipf & Welling 2017: hidden width 16, dropout 0.5) rather than the
    manuscript's gamma-band + PLV tuning, so a run reflects an untuned,
    off-the-shelf GCN baseline rather than settings borrowed from an
    unrelated feature combination.
    """

    classifier: str = "mlp"
    edge_mode: str = "weighted"
    nodes: int = EEGGCN1_NODES
    input_features: int = EEGGCN1_INPUT_FEATURES
    gcn_width: int = 16
    output_classes: int = 2
    dropout: float = 0.5
    leaky_relu_slope: float = 0.01

    def __post_init__(self) -> None:
        if self.classifier not in ("mlp", "linear"):
            raise ValueError("classifier must be mlp or linear")
        if self.edge_mode not in ("weighted", "self_only"):
            raise ValueError("edge_mode must be weighted or self_only")
        channel_values = (
            self.nodes,
            self.input_features,
            self.gcn_width,
            self.output_classes,
        )
        if any(value <= 0 for value in channel_values):
            raise ValueError("nodes, input_features, gcn_width, and output_classes must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        if self.leaky_relu_slope < 0:
            raise ValueError("leaky_relu_slope cannot be negative")


class EEGGCN1(nn.Module):
    """Classify fixed-size sparse graphs using one weighted GCNConv layer."""

    def __init__(self, config: EEGGCN1Config | None = None) -> None:
        super().__init__()
        self.config = config or EEGGCN1Config()
        self.convolution = GCNConv(self.config.input_features, self.config.gcn_width)
        self.batch_norm = nn.BatchNorm1d(self.config.gcn_width)
        self.activation = nn.LeakyReLU(negative_slope=self.config.leaky_relu_slope)
        self.dropout = nn.Dropout(self.config.dropout)
        flat_width = self.config.nodes * self.config.gcn_width
        self.classifier_hidden = (
            nn.Linear(flat_width, self.config.nodes)
            if self.config.classifier == "mlp" else None
        )
        self.classifier_output = nn.Linear(
            self.config.nodes if self.config.classifier == "mlp" else flat_width,
            self.config.output_classes,
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return graph-level log-probabilities shaped ``(graphs, classes)``."""

        batch = self._validate_inputs(x, edge_index, edge_weight, batch)

        if self.config.edge_mode == "self_only":
            # Identity adjacency: each electrode receives only its own features.
            # Explicit unit loops retain the same GCN parameters and normalization.
            indices = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
            edge_index = torch.stack((indices, indices))
            edge_weight = x.new_ones(x.shape[0])
        hidden = self.convolution(x, edge_index, edge_weight)
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(self.activation(hidden))

        dense_hidden, mask = to_dense_batch(
            hidden, batch, max_num_nodes=self.config.nodes
        )
        if not bool(mask.all()):
            raise ValueError(f"Every graph must contain exactly {self.config.nodes} nodes")
        batch_size = dense_hidden.shape[0]
        flat = dense_hidden.reshape(batch_size, self.config.nodes * self.config.gcn_width)

        if self.classifier_hidden is not None:
            flat = self.activation(self.classifier_hidden(flat))
            flat = self.dropout(flat)
        logits = self.classifier_output(flat)
        return self.log_softmax(logits)

    def _validate_inputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        """Validate the sparse tensors required by the EEGGCN1 forward pass."""

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a node-feature tensor")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("edge_index must be an edge-endpoint tensor")
        if not isinstance(edge_weight, torch.Tensor):
            raise TypeError("edge_weight must be an edge-weight tensor")
        if x.ndim != 2 or x.shape[1] != self.config.input_features:
            raise ValueError(f"x must have shape (total_nodes, {self.config.input_features})")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, total_edges)")
        if not torch.is_floating_point(x):
            raise TypeError("x must use a floating-point dtype")
        if edge_index.dtype not in (torch.int64, torch.int32):
            raise TypeError("edge_index must use an integer dtype")
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must have shape (total_edges,) matching edge_index")
        if not torch.is_floating_point(edge_weight):
            raise TypeError("edge_weight must use a floating-point dtype")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        if not torch.isfinite(edge_weight).all():
            raise ValueError("edge_weight must contain only finite values")

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        elif not isinstance(batch, torch.Tensor):
            raise TypeError("batch must be a graph-assignment tensor or None")
        elif batch.ndim != 1 or batch.shape[0] != x.shape[0]:
            raise ValueError("batch must have shape (total_nodes,) matching x")
        elif batch.dtype not in (torch.int64, torch.int32):
            raise TypeError("batch must use an integer dtype")

        if x.shape[0] % self.config.nodes != 0:
            raise ValueError(
                f"Total node count {x.shape[0]} is not a multiple of {self.config.nodes}"
            )
        return batch.to(torch.long)
