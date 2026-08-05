"""Physiological tokenization front-end (CFSPMNet paper, Eq. 3).

Ported from ``temp/cfspmnet_frsmamba_model.py``.
"""

from __future__ import annotations

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class DepthwiseSeparableTokenFusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=padding,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MultiScalePhysiologicalTokenizer(nn.Module):
    def __init__(
        self,
        f1=10,
        D=3,
        pooling_size1=8,
        pooling_size2=8,
        dropout_rate=0.3,
        number_channel=22,
        emb_size=40,
        temporal_kernel_sizes=(36, 24, 18),
        fusion_kernel_size=16,
    ):
        super().__init__()

        if len(temporal_kernel_sizes) != 3:
            raise ValueError("temporal_kernel_sizes must contain exactly 3 kernel sizes.")

        temporal_channels = f1 * len(temporal_kernel_sizes)
        fusion_channels = temporal_channels * D

        self.temporal_branches = nn.ModuleList(
            [
                nn.Conv2d(1, f1, kernel_size=(1, kernel), padding="same", bias=False)
                for kernel in temporal_kernel_sizes
            ]
        )
        self.activation = nn.ELU()
        self.temporal_bn = nn.BatchNorm2d(temporal_channels)

        self.spatial_conv = nn.Conv2d(
            temporal_channels,
            fusion_channels,
            kernel_size=(number_channel, 1),
            groups=temporal_channels,
            padding="valid",
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(fusion_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, pooling_size1), stride=(1, pooling_size1))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fusion_conv = DepthwiseSeparableTokenFusionConv(
            fusion_channels,
            fusion_channels,
            kernel_size=(1, fusion_kernel_size),
            padding="same",
        )
        self.fusion_bn = nn.BatchNorm2d(fusion_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, pooling_size2), stride=(1, pooling_size2))
        self.dropout2 = nn.Dropout(dropout_rate)

        self.sequence_projection = Rearrange("b e h w -> b (h w) e")
        self.embedding_projection = (
            nn.Identity() if fusion_channels == emb_size else nn.Linear(fusion_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([branch(x) for branch in self.temporal_branches], dim=1)
        x = self.temporal_bn(self.activation(x))

        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.fusion_conv(x)
        x = self.fusion_bn(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.sequence_projection(x)
        x = self.embedding_projection(x)
        return x


class LearnablePositionEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        x = x + self.encoding[:, : x.shape[1], :].to(x.device)
        return self.dropout(x)
