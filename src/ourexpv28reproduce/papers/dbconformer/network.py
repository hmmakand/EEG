"""DBConformer, ported from models/DBConformer.py with the scripts' configuration
(positional embeddings, channel attention, both branches, no gate).

    temporal branch: Conv1d 1x1 (channels -> 40) + BN -> depthwise Conv1d (kernel 63) + BN -> GELU
                     -> AvgPool (125, non-overlapping) -> Dropout 0.5 -> patches [P, 40] + positions
                     -> transformer (pre-LayerNorm, 10 heads) -> mean over the patches
    spatial branch:  per channel: Conv1d (1 -> 16, kernel 25, stride 5) -> ELU -> mean over time -> Linear 40
                     -> channel tokens [C, 40] + positions -> transformer -> channel attention (Linear, tanh,
                     Linear, softmax over the channels) -> weighted sum
    classifier:      [temporal; spatial] (80) -> Linear 64 -> ELU -> Dropout 0.5 -> Linear 32 -> ELU
                     -> Dropout 0.3 -> Linear

Kept as the code does it: the temporal patch embedding's convolutions are
initialised with a truncated normal (std 0.01) and its batch norms with 1 and
0; everything else has PyTorch's default initialisation; attention scores
are divided by the square root of the embedding size (40). The code drops
the last time point before pooling, except for its 750-sample dataset, so
that 1,001 samples give 8 patches; our samples are not one sample too long,
so nothing is dropped (as for its 750-sample dataset).
"""
import torch
import torch.nn.functional as F
from torch import nn

from .settings import ModelSettings


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


class EncoderBlock(nn.Module):
    def __init__(self, size: int, heads: int, dropout: float, expansion: int):
        super().__init__()
        self.attention = nn.Sequential(nn.LayerNorm(size), Attention(size, heads, dropout), nn.Dropout(dropout))
        self.feed_forward = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, expansion * size), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(expansion * size, size), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(x)
        return x + self.feed_forward(x)


def encoder(depth: int, s: ModelSettings):
    return nn.Sequential(*[EncoderBlock(s.embedding, s.heads, s.dropout, s.forward_expansion) for _ in range(depth)])


class DBConformer(nn.Module):
    """Input [batch, channels, time] at 250 Hz; output logits [batch, classes]."""

    def __init__(self, channels: int, samples: int, depth: int, classes: int = 2,
                 settings: ModelSettings = ModelSettings()):
        super().__init__()
        d = settings.embedding
        self.patches = samples // settings.patch
        if self.patches < 1:
            raise ValueError(f"DBConformer needs at least {settings.patch} time samples")
        # temporal patch embedding (Stem with radix 1)
        self.pointwise = nn.Conv1d(channels, d, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm1d(d)
        self.depthwise = nn.Conv1d(d, d, settings.temporal_kernel, padding=settings.temporal_kernel // 2, groups=d,
                                   bias=False)
        self.depthwise_bn = nn.BatchNorm1d(d)
        self.pool = nn.AvgPool1d(settings.patch, settings.patch)
        self.patch_drop = nn.Dropout(settings.dropout)
        for module in (self.pointwise, self.depthwise):
            nn.init.trunc_normal_(module.weight, std=0.01)
        # spatial patch embedding
        self.channel_embedding = nn.Sequential(
            nn.Conv1d(1, settings.spatial_filters, settings.spatial_kernel, stride=settings.spatial_stride,
                      padding=settings.spatial_kernel // 2), nn.ELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(settings.spatial_filters, d))
        self.position_temporal = nn.Parameter(torch.randn(1, self.patches, d))
        self.position_spatial = nn.Parameter(torch.randn(1, channels, d))
        self.temporal_encoder = encoder(depth, settings)
        self.spatial_encoder = encoder(depth, settings)
        (h1, h2), (d1, d2) = settings.hidden, settings.hidden_dropout
        self.classifier = nn.Sequential(nn.Linear(2 * d, h1), nn.ELU(), nn.Dropout(d1), nn.Linear(h1, h2), nn.ELU(),
                                        nn.Dropout(d2), nn.Linear(h2, classes))
        self.channel_attention = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def features(self, x):
        b, c, t = x.shape
        temporal = self.depthwise_bn(self.depthwise(self.pointwise_bn(self.pointwise(x))))
        temporal = self.patch_drop(self.pool(F.gelu(temporal))).transpose(1, 2)          # [b, P, d]
        spatial = self.channel_embedding(x.reshape(b * c, 1, t)).view(b, c, -1)         # [b, C, d]
        temporal = self.temporal_encoder(temporal + self.position_temporal).mean(dim=1)
        spatial = self.spatial_encoder(spatial + self.position_spatial)
        weights = torch.softmax(self.channel_attention(spatial), dim=1)
        return torch.cat([temporal, (weights * spatial).sum(dim=1)], dim=-1)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(channels: int, samples: int, depth: int, classes: int = 2,
                settings: ModelSettings = ModelSettings()) -> DBConformer:
    return DBConformer(channels, samples, depth, classes, settings)
