"""EEG Conformer, ported from conformer.py of the published code.

    convolution module: Conv (1 x 25, 40) -> Conv (channels x 1, 40) -> BatchNorm -> ELU
                        -> AvgPool (1 x 75, stride 15) -> Dropout 0.5 -> Conv 1 x 1 -> tokens [time, 40]
    self-attention:     6 x (LayerNorm -> 10-head attention -> Dropout, residual;
                             LayerNorm -> Linear 160 -> GELU -> Dropout -> Linear 40 -> Dropout, residual)
    classifier:         flatten -> Linear 256 -> ELU -> Dropout 0.5 -> Linear 32 -> ELU -> Dropout 0.3 -> Linear

The spatial convolution is computed as a tensor product (SpatialConv): the
same parameters and the same result as nn.Conv2d (checked, with gradients),
about four times faster, and without the multi-gigabyte workspace that cuDNN
takes on the first call for 63 x 1,708 inputs.

Kept as the code does it: the attention scores are divided by the square root
of the token size (40), not of the head size; PyTorch's default
initialisation; biases everywhere. The code's unused average-pooling head
(`clshead`) is left out, and the last layer has as many outputs as classes
(the code fixes it at 4).
"""
import torch
import torch.nn.functional as F
from torch import nn

from .settings import ModelSettings


class SpatialConv(nn.Conv2d):
    """nn.Conv2d(k, k, (channels, 1)) on [b, k, channels, t], computed with einsum."""

    def forward(self, x):
        out = torch.einsum("oic,bict->bot", self.weight[..., 0], x).unsqueeze(2)
        return out + self.bias.view(1, -1, 1, 1)


class Attention(nn.Module):
    def __init__(self, size: int, heads: int, dropout: float):
        super().__init__()
        self.size, self.heads = size, heads
        self.keys, self.queries, self.values = nn.Linear(size, size), nn.Linear(size, size), nn.Linear(size, size)
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(size, size)

    def forward(self, x):
        split = lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)   # b h n d
        q, k, v = split(self.queries(x)), split(self.keys(x)), split(self.values(x))
        attention = self.drop(F.softmax(q @ k.transpose(-1, -2) / self.size ** 0.5, dim=-1))
        out = (attention @ v).transpose(1, 2).reshape(x.shape)
        return self.projection(out)


class EncoderBlock(nn.Module):
    def __init__(self, size: int, heads: int, dropout: float, expansion: int):
        super().__init__()
        self.attention = nn.Sequential(nn.LayerNorm(size), Attention(size, heads, dropout), nn.Dropout(dropout))
        self.feed_forward = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, expansion * size), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(expansion * size, size), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(x)
        return x + self.feed_forward(x)


class Conformer(nn.Module):
    """Input [batch, channels, time] at 250 Hz; output logits [batch, classes]."""

    def __init__(self, channels: int, samples: int, classes: int = 2, settings: ModelSettings = ModelSettings()):
        super().__init__()
        k = settings.kernels
        self.convolution = nn.Sequential(
            nn.Conv2d(1, k, (1, settings.temporal_kernel)), SpatialConv(k, k, (channels, 1)), nn.BatchNorm2d(k),
            nn.ELU(), nn.AvgPool2d((1, settings.pool), (1, settings.pool_stride)), nn.Dropout(settings.dropout),
            nn.Conv2d(k, k, (1, 1)))
        self.encoder = nn.Sequential(*[EncoderBlock(k, settings.heads, settings.dropout, settings.forward_expansion)
                                       for _ in range(settings.depth)])
        tokens = (samples - settings.temporal_kernel + 1 - settings.pool) // settings.pool_stride + 1
        if tokens < 1:
            raise ValueError("The sample is too short for the convolution module")
        (h1, h2), (d1, d2) = settings.hidden, settings.hidden_dropout
        self.classifier = nn.Sequential(nn.Linear(tokens * k, h1), nn.ELU(), nn.Dropout(d1), nn.Linear(h1, h2),
                                        nn.ELU(), nn.Dropout(d2), nn.Linear(h2, classes))

    def tokens(self, x):
        x = self.convolution(x.unsqueeze(1))                      # [b, k, 1, tokens]
        return self.encoder(x.flatten(2).transpose(1, 2))         # [b, tokens, k]

    def forward(self, x):
        return self.classifier(self.tokens(x).flatten(1))


def build_model(channels: int, samples: int, classes: int = 2, settings: ModelSettings = ModelSettings()) -> Conformer:
    return Conformer(channels, samples, classes, settings)
