"""EEGNet in PyTorch, ported from EEGNet() in original/EEGModels.py (Keras).

Kept as in the Keras code, where PyTorch's defaults differ:
  - no bias in the convolutions; a bias in the dense layer;
  - 'same' padding (PyTorch splits an even kernel's padding as Keras does);
  - batch normalisation with Keras's momentum 0.99 (PyTorch: 0.01) and epsilon 1e-3;
  - Glorot-uniform initialisation with the fan-in and fan-out Keras computes for each kernel shape;
  - max-norm constraints, applied after every optimiser step as Keras does: each spatial filter to norm 1,
    each output unit of the dense layer to norm 0.25 (keras.constraints.max_norm).
The softmax is left to the loss (cross-entropy on the logits).
"""
import math

import torch
from torch import nn

from .settings import ModelSettings


def glorot_(weight, fan_in: int, fan_out: int) -> None:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(weight, -limit, limit)


def max_norm_(weight, max_value: float) -> None:
    """keras.constraints.max_norm over every dimension but the first: w * clip(n, 0, max) / (1e-7 + n)."""
    norms = weight.flatten(1).norm(dim=1).view(-1, *[1] * (weight.dim() - 1))
    weight.mul_(norms.clamp(max=max_value) / (1e-7 + norms))


class EEGNet(nn.Module):
    """Input [batch, channels, time] at 128 Hz; output logits [batch, classes]."""

    def __init__(self, channels: int, samples: int, classes: int = 2, dropout: float = 0.5,
                 settings: ModelSettings = ModelSettings()):
        super().__init__()
        f1, d, f2, k = settings.temporal_filters, settings.depth, settings.pointwise_filters, settings.kernel
        self.settings = settings
        norm = dict(momentum=0.01, eps=1e-3)
        self.temporal = nn.Conv2d(1, f1, (1, k), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1, **norm)
        self.spatial = nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False)       # DepthwiseConv2D
        self.bn2 = nn.BatchNorm2d(f1 * d, **norm)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        s = settings.separable_kernel
        self.separable_depthwise = nn.Conv2d(f1 * d, f1 * d, (1, s), padding="same", groups=f1 * d, bias=False)
        self.separable_pointwise = nn.Conv2d(f1 * d, f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2, **norm)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        if samples // 4 // 8 < 1:
            raise ValueError("EEGNet needs at least 32 time samples")
        self.dense = nn.Linear(f2 * (samples // 4 // 8), classes)
        self.elu = nn.ELU()

        # Keras kernel shapes: Conv2D (1, k, 1, F1); DepthwiseConv2D (C, 1, F1, D);
        # SeparableConv2D depthwise (1, s, F1*D, 1) and pointwise (1, 1, F1*D, F2); Dense (in, classes).
        glorot_(self.temporal.weight, k, k * f1)
        glorot_(self.spatial.weight, channels * f1, channels * d)
        glorot_(self.separable_depthwise.weight, s * f1 * d, s)
        glorot_(self.separable_pointwise.weight, f1 * d, f2)
        glorot_(self.dense.weight, self.dense.in_features, classes)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x):
        x = self.bn1(self.temporal(x.unsqueeze(1)))
        x = self.dropout1(self.pool1(self.elu(self.bn2(self.spatial(x)))))
        x = self.bn3(self.separable_pointwise(self.separable_depthwise(x)))
        x = self.dropout2(self.pool2(self.elu(x)))
        return self.dense(x.flatten(1))

    @torch.no_grad()
    def apply_constraints(self) -> None:
        max_norm_(self.spatial.weight, self.settings.depthwise_max_norm)
        max_norm_(self.dense.weight, self.settings.dense_max_norm)


def build_model(channels: int, samples: int, dropout: float, classes: int = 2,
                settings: ModelSettings = ModelSettings()) -> EEGNet:
    return EEGNet(channels, samples, classes, dropout, settings)
