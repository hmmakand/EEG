"""GATv2 model for BCI Competition IV 2a left/right motor-imagery classification.

Ported as-is from src/ourexperimentversionseven/kaggleeeg.ipynb (cell 1c18b945),
the model class actually used to produce BCI_IV_2a_GAT_Results.json.

One deviation from the notebook: there, `in_channels` was read from the
notebook's global `band` variable (`len(band) - 1`). Here it's an explicit
constructor argument defaulting to 8 (the same value `band = range(8, 41, 4)`
always produced) so the class doesn't depend on module-level globals.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool


class GAT(nn.Module):
    def __init__(self, hidden_channels, heads, in_channels: int = 8):
        super().__init__()

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)

        self.gn1 = GraphNorm(hidden_channels * heads)
        self.gn2 = GraphNorm(hidden_channels * heads)
        self.gn3 = GraphNorm(hidden_channels * heads)

        self.lin = nn.Linear(hidden_channels * heads, 2)  # binary classification

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.gn1(x, batch)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.gn2(x, batch)

        x = self.conv3(x, edge_index)
        x = self.gn3(x, batch)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.50, training=self.training)
        x = self.lin(x)

        return x
