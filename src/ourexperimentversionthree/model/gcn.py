"""EEGGCN1: manuscript-faithful dense graph convolutional classifier.

Architecture from Zhang et al. (2026), Table 2: one ``DenseGCNConv`` over a
dense adjacency, followed by BatchNorm/LeakyReLU/dropout, a flatten of all
node embeddings into one vector, and a two-layer fully connected classifier
ending in log-softmax. Unlike ``torch_geometric``'s sparse ``GCNConv``, this
takes dense ``x: (B, N, F)`` node features and ``adj: (B, N, N)`` adjacency
tensors directly -- there is no ``edge_index``/``edge_attr``/PyG ``Data``
object involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import DenseGCNConv


EEGGCN1_NODES = 29
EEGGCN1_INPUT_FEATURES = 11
"""Node count and feature width of the manuscript broadcast-11 dataset."""


@dataclass(frozen=True)
class EEGGCN1Config:
    """Architecture options for the manuscript's EEGGCN1 classifier."""

    nodes: int = EEGGCN1_NODES
    input_features: int = EEGGCN1_INPUT_FEATURES
    gcn_width: int = 14
    output_classes: int = 2
    dropout: float = 0.4
    leaky_relu_slope: float = 0.01

    def __post_init__(self) -> None:
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
    """Classify a fixed-size dense graph using one DenseGCNConv layer."""

    def __init__(self, config: EEGGCN1Config | None = None) -> None:
        super().__init__()
        self.config = config or EEGGCN1Config()
        self.convolution = DenseGCNConv(
            self.config.input_features, self.config.gcn_width
        )
        self.batch_norm = nn.BatchNorm1d(self.config.gcn_width)
        self.activation = nn.LeakyReLU(negative_slope=self.config.leaky_relu_slope)
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier_hidden = nn.Linear(
            self.config.nodes * self.config.gcn_width, self.config.nodes
        )
        self.classifier_output = nn.Linear(
            self.config.nodes, self.config.output_classes
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Return graph-level log-probabilities shaped ``(graphs, classes)``."""

        self._validate_inputs(x, adj)
        batch_size = x.shape[0]

        hidden = self.convolution(x, adj)
        hidden = hidden.reshape(batch_size * self.config.nodes, self.config.gcn_width)
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(self.activation(hidden))
        hidden = hidden.reshape(batch_size, self.config.nodes * self.config.gcn_width)

        hidden = self.activation(self.classifier_hidden(hidden))
        hidden = self.dropout(hidden)
        logits = self.classifier_output(hidden)
        return self.log_softmax(logits)

    def _validate_inputs(self, x: torch.Tensor, adj: torch.Tensor) -> None:
        """Validate the dense tensors required by the EEGGCN1 forward pass."""

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a node-feature tensor")
        if not isinstance(adj, torch.Tensor):
            raise TypeError("adj must be a dense adjacency tensor")
        if x.ndim != 3 or x.shape[1:] != (self.config.nodes, self.config.input_features):
            raise ValueError(
                f"x must have shape (batch, {self.config.nodes}, {self.config.input_features})"
            )
        if adj.ndim != 3 or adj.shape[1:] != (self.config.nodes, self.config.nodes):
            raise ValueError(
                f"adj must have shape (batch, {self.config.nodes}, {self.config.nodes})"
            )
        if adj.shape[0] != x.shape[0]:
            raise ValueError("x and adj must share the same batch size")
        if not torch.is_floating_point(x):
            raise TypeError("x must use a floating-point dtype")
        if not torch.is_floating_point(adj):
            raise TypeError("adj must use a floating-point dtype")
        if adj.dtype != x.dtype:
            raise TypeError("x and adj must use the same dtype")
        if adj.device != x.device:
            raise ValueError("x and adj must be on the same device")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        if not torch.isfinite(adj).all():
            raise ValueError("adj must contain only finite values")
