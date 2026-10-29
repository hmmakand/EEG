"""M-ShallowConvNet, from model/mshallowconvnet.py and model/layers.py of the published code.

    temporal convolution (1 x 30, 24 kernels, 'same') -> spatial convolution (channels x 1, 24 kernels)
    -> square -> average pool (1 x 75, stride 22) -> log (clamped at 1e-6) -> batch normalisation
    -> flatten -> dropout 0.5 -> dense

Kept as the code does it:
  - both convolutions have a bias (PyTorch's default) and PyTorch's default initialisation;
  - their weights are renormed to a maximum norm of 2 per kernel before every forward pass
    (Conv2dWithConstraint);
  - the dense layer is a LazyLinearWithConstraint. PyTorch turns a lazy layer into a plain Linear at its
    first call, so its max-norm of 0.5 is applied once, at the first forward pass, and never again. Here
    it is applied once after construction, which is the same because nothing changes the weights in
    between.
"""
import torch
from torch import nn

from .settings import ModelSettings

SAMPLE_RATE = 250


class MShallowConvNet(nn.Module):
    """Input [batch, channels, time] at 250 Hz; output logits [batch, classes]."""

    def __init__(self, channels: int, samples: int, classes: int = 2, settings: ModelSettings = ModelSettings()):
        super().__init__()
        self.settings = settings
        d = settings.kernels
        kernel = int(SAMPLE_RATE * settings.temporal_kernel_seconds)
        pool = int(SAMPLE_RATE * settings.pool_seconds)
        stride = int(SAMPLE_RATE * settings.pool_seconds * (1 - settings.pool_overlap))
        self.temporal = nn.Conv2d(1, d, (1, kernel), padding="same")
        self.spatial = nn.Conv2d(d, d, (channels, 1))
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        self.bn = nn.BatchNorm2d(d)
        self.dropout = nn.Dropout(settings.dropout)
        steps = (samples - pool) // stride + 1
        if steps < 1:
            raise ValueError(f"M-ShallowConvNet needs at least {pool} time samples")
        self.dense = nn.Linear(d * steps, classes)
        with torch.no_grad():
            self.dense.weight.data = torch.renorm(self.dense.weight.data, p=2, dim=0, maxnorm=settings.dense_max_norm)

    def forward(self, x):
        for conv in (self.temporal, self.spatial):
            conv.weight.data = torch.renorm(conv.weight.data, p=2, dim=0, maxnorm=self.settings.conv_max_norm)
        x = self.spatial(self.temporal(x.unsqueeze(1)))
        x = torch.log(torch.clamp(self.pool(torch.square(x)), min=self.settings.log_floor))
        return self.dense(self.dropout(self.bn(x).flatten(1)))


def build_model(channels: int, samples: int, classes: int = 2,
                settings: ModelSettings = ModelSettings()) -> MShallowConvNet:
    return MShallowConvNet(channels, samples, classes, settings)
