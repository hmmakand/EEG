"""SlimSeiz's network: the OneDCNN class of Loop_ADDNet_kFold.ipynb (the paper's Fig. 2).

    Conv1d (k channels -> 16, kernel 21) ReLU, MaxPool 8
    residual: [Conv1d 1x1] + [Conv1d 11, ReLU, Conv1d 3, ReLU] (16), MaxPool 4
    residual: [Conv1d 1x1 stride 2] + [Conv1d 5 stride 2, ReLU, Conv1d 3, ReLU] (32)
    Mamba block on the time steps (RMSNorm before it, residual around it)
    global average pooling -> the 32 features (used by the contrastive loss) -> Linear 2

The Mamba block is the code's pure-PyTorch one (from mamba-minimal):
expansion 2, state 16, Delta rank ceil(32 / 16) = 2, depthwise causal
convolution of kernel 3, and a sequential selective scan. Ported unchanged
except for einops, which is replaced by the same tensor operations.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class MambaBlock(nn.Module):
    def __init__(self, size: int, state: int = 16, expansion: int = 2, kernel: int = 3):
        super().__init__()
        self.inner = inner = size * expansion
        self.rank = math.ceil(size / 16)
        self.state = state
        self.in_proj = nn.Linear(size, 2 * inner)
        self.conv1d = nn.Conv1d(inner, inner, kernel, groups=inner, padding=kernel - 1)
        self.x_proj = nn.Linear(inner, self.rank + 2 * state, bias=False)
        self.dt_proj = nn.Linear(self.rank, inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state + 1, dtype=torch.float).repeat(inner, 1)))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, size)

    def forward(self, x):                                            # [b, l, size]
        length = x.shape[1]
        x, gate = self.in_proj(x).split(self.inner, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :length].transpose(1, 2))
        return self.out_proj(self.ssm(x) * F.silu(gate))

    def ssm(self, u):
        a = -torch.exp(self.A_log.float())                           # [inner, state]
        delta, b, c = self.x_proj(u).split([self.rank, self.state, self.state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))                      # [b, l, inner]
        delta_a = torch.exp(delta.unsqueeze(-1) * a)                 # [b, l, inner, state]
        delta_b_u = delta.unsqueeze(-1) * b.unsqueeze(2) * u.unsqueeze(-1)
        h = u.new_zeros(u.shape[0], self.inner, self.state)
        outputs = []
        for i in range(u.shape[1]):
            h = delta_a[:, i] * h + delta_b_u[:, i]
            outputs.append((h * c[:, i].unsqueeze(1)).sum(-1))
        return torch.stack(outputs, dim=1) + u * self.D.float()


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SlimSeiz(nn.Module):
    """Input [batch, channels, time]; output (logits [batch, 2], features [batch, 32])."""

    def __init__(self, channels: int, classes: int = 2):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(channels, 16, 21, padding=10), nn.ReLU(), nn.MaxPool1d(8, 8))
        self.conv2_1 = nn.Sequential(nn.Conv1d(16, 16, 1), nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv1d(16, 16, 11, padding=5), nn.ReLU(), nn.Conv1d(16, 16, 3, padding=1),
                                     nn.ReLU())
        self.pool3 = nn.MaxPool1d(4, 4)
        self.conv4_1 = nn.Sequential(nn.Conv1d(16, 32, 1, stride=2), nn.ReLU())
        self.conv4_2 = nn.Sequential(nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
                                     nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        self.mixer = MambaBlock(32)
        self.norm = RMSNorm(32)
        self.classifier = nn.Sequential(nn.Linear(32, classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool3(self.conv2_1(x) + self.conv2_2(x))
        x = (self.conv4_1(x) + self.conv4_2(x)).transpose(1, 2)        # [b, steps, 32]
        x = self.mixer(self.norm(x)) + x
        features = x.mean(dim=1)                                       # AdaptiveAvgPool1d(1)
        return self.classifier(features), features


def build_model(channels: int, classes: int = 2) -> SlimSeiz:
    return SlimSeiz(channels, classes)
