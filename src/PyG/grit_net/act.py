"""Activation registry.

Replaces `torch_geometric.graphgym.register.act_dict` + `grit/act/example.py`.
Each entry is a *class/factory* (call it to get an nn.Module), matching how the
original code uses `register.act_dict[act]()`.
"""
from functools import partial

import torch
import torch.nn as nn


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        return x * torch.sigmoid(x)


class SignedSqrt(nn.Module):
    """sign(x) * sqrt(|x|) — used inside GRIT attention."""

    def forward(self, x):
        return torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))


act_dict = {
    'relu': nn.ReLU,
    'selu': nn.SELU,
    'prelu': nn.PReLU,
    'elu': nn.ELU,
    'lrelu_01': partial(nn.LeakyReLU, 0.1),
    'lrelu_025': partial(nn.LeakyReLU, 0.25),
    'lrelu_02': partial(nn.LeakyReLU, 0.2),
    'lrelu_03': partial(nn.LeakyReLU, 0.3),
    'lrelu_05': partial(nn.LeakyReLU, 0.5),
    'gelu': nn.GELU,
    'swish': SWISH,
    'signed_sqrt': SignedSqrt,
}
