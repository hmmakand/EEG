"""MSCFormer, ported from MSCFormer_model.py of the published code.

    three branches, temporal kernels 85, 65 and 45:
        Conv (1 x K, 16, 'same') -> depthwise spatial Conv (channels x 1, groups 16) -> BatchNorm -> ELU
        -> AvgPool (1 x 44) -> Dropout
    concatenate the branches -> tokens [time / 44, 48]
    prepend a class token -> multiply by sqrt(48) -> + learned position embedding -> Dropout 0.1
    5 transformer layers (8 heads, post-LayerNorm: LayerNorm(x + Dropout(f(x))), feed-forward 192, GELU)
    the class token's output -> Dropout 0.25 -> Linear

The depthwise spatial convolution is computed as a tensor product
(DepthwiseSpatialConv): the same parameters and the same result as the
grouped nn.Conv2d (checked, with gradients), several times faster, because
PyTorch runs that grouped convolution with a slow generic kernel.

Kept as the code does it: the class token is a zero vector, so after the
position embedding it equals the learned first position (the paper calls it
a learnable vector; the two are the same); attention scores are divided by
the square root of the token size (48); PyTorch's default initialisation.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .settings import ModelSettings


class DepthwiseSpatialConv(nn.Conv2d):
    """nn.Conv2d(f, f, (channels, 1), groups=f) on [b, f, channels, t], computed with einsum."""

    def forward(self, x):
        out = torch.einsum("fc,bfct->bft", self.weight[:, 0, :, 0], x).unsqueeze(2)
        return out + self.bias.view(1, -1, 1, 1)


class Attention(nn.Module):
    def __init__(self, size: int, heads: int, dropout: float):
        super().__init__()
        self.size, self.heads = size, heads
        self.keys, self.queries, self.values = nn.Linear(size, size), nn.Linear(size, size), nn.Linear(size, size)
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(size, size)

    def forward(self, x):
        split = lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)
        q, k, v = split(self.queries(x)), split(self.keys(x)), split(self.values(x))
        attention = self.drop(F.softmax(q @ k.transpose(-1, -2) / self.size ** 0.5, dim=-1))
        return self.projection((attention @ v).transpose(1, 2).reshape(x.shape))


class PostNormResidual(nn.Module):
    """ResidualAdd of the code: LayerNorm(Dropout(f(x)) + x)."""

    def __init__(self, function, size: int, dropout: float):
        super().__init__()
        self.function, self.drop, self.norm = function, nn.Dropout(dropout), nn.LayerNorm(size)

    def forward(self, x):
        return self.norm(self.drop(self.function(x)) + x)


class MSCFormer(nn.Module):
    """Input [batch, channels, time] at 250 Hz; output logits [batch, classes]."""

    def __init__(self, channels: int, samples: int, classes: int = 2, branch_dropout: float = 0.5,
                 settings: ModelSettings = ModelSettings()):
        super().__init__()
        f1 = settings.temporal_filters
        self.size = size = 3 * f1
        self.branches = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, f1, (1, k), padding="same"), DepthwiseSpatialConv(f1, f1, (channels, 1), groups=f1),
            nn.BatchNorm2d(f1), nn.ELU(), nn.AvgPool2d((1, settings.pool)), nn.Dropout(branch_dropout))
            for k in settings.kernels])
        tokens = samples // settings.pool
        if not 1 <= tokens < settings.max_tokens:
            raise ValueError(f"{samples} samples give {tokens} tokens; 1 to {settings.max_tokens - 1} are possible")
        self.position = nn.Parameter(torch.randn(1, settings.max_tokens, size))
        self.position_drop = nn.Dropout(settings.position_dropout)
        d = settings.dropout
        self.encoder = nn.Sequential(*[nn.Sequential(
            PostNormResidual(Attention(size, settings.heads, d), size, d),
            PostNormResidual(nn.Sequential(nn.Linear(size, 4 * size), nn.GELU(), nn.Dropout(d),
                                           nn.Linear(4 * size, size)), size, d))
            for _ in range(settings.depth)])
        self.classifier = nn.Sequential(nn.Dropout(settings.classifier_dropout), nn.Linear(size, classes))

    def features(self, x):
        x = x.unsqueeze(1)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)          # [b, 48, 1, tokens]
        x = x.flatten(2).transpose(1, 2)                                       # [b, tokens, 48]
        x = torch.cat([x.new_zeros(x.shape[0], 1, self.size), x], dim=1) * math.sqrt(self.size)
        x = self.position_drop(x + self.position[:, :x.shape[1]])
        return self.encoder(x)[:, 0]

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(channels: int, samples: int, classes: int = 2, branch_dropout: float = 0.5,
                settings: ModelSettings = ModelSettings()) -> MSCFormer:
    return MSCFormer(channels, samples, classes, branch_dropout, settings)
