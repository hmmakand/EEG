"""Composition of the node feature encoder with positional-encoding encoders.

Original: `graphgps/encoder/composed_encoders.py`, which registered names like
'LinearNode+LapPE', 'LinearNode+RWSE', 'LinearNode+LapPE+RWSE'. Here the same
Concat2 / Concat3 logic is kept, and the combination is picked from which
`cfg.posenc_<Name>.enable` flags are set.

Dimension split (same as original):
    no PE           : x -> Linear(dim_in, dim_emb)
    one PE (dim p)  : x -> Linear(dim_in, dim_emb - p)  ||  PE(p)
    two PEs (p2,p3) : x -> Linear(dim_in, dim_emb - p2 - p3) || PE2(p2) || PE3(p3)
    EquivStableLapPE: x -> Linear(dim_in, dim_emb), PE kept separately
"""
import torch

from .equivstable_laplace_pos_encoder import EquivStableLapPENodeEncoder
from .kernel_pos_encoder import RWSENodeEncoder
from .laplace_pos_encoder import LapPENodeEncoder
from .linear_node_encoder import LinearNodeEncoder

# Positional Encoding node encoders (subset of the original `pe_encs`).
pe_encs = {'LapPE': LapPENodeEncoder,
           'RWSE': RWSENodeEncoder}


class Concat2NodeEncoder(torch.nn.Module):
    """Encoder that concatenates two node encoders."""

    def __init__(self, dim_in, dim_emb, pe_name, pecfg):
        super().__init__()
        self.encoder1 = LinearNodeEncoder(dim_in, dim_emb - pecfg.dim_pe)
        self.encoder2 = pe_encs[pe_name](dim_in, dim_emb, pecfg, expand_x=False)

    def forward(self, batch):
        batch = self.encoder1(batch)
        batch = self.encoder2(batch)
        return batch


class Concat3NodeEncoder(torch.nn.Module):
    """Encoder that concatenates three node encoders."""

    def __init__(self, dim_in, dim_emb, pe_names, pecfgs):
        super().__init__()
        (name2, name3), (cfg2, cfg3) = pe_names, pecfgs
        self.encoder1 = LinearNodeEncoder(
            dim_in, dim_emb - cfg2.dim_pe - cfg3.dim_pe)
        self.encoder2 = pe_encs[name2](dim_in, dim_emb - cfg3.dim_pe, cfg2,
                                       expand_x=False)
        self.encoder3 = pe_encs[name3](dim_in, dim_emb, cfg3, expand_x=False)

    def forward(self, batch):
        batch = self.encoder1(batch)
        batch = self.encoder2(batch)
        batch = self.encoder3(batch)
        return batch


class EquivStableConcatNodeEncoder(torch.nn.Module):
    """Special case: node feats and EquivStableLapPE are not concatenated."""

    def __init__(self, dim_in, dim_emb, pecfg):
        super().__init__()
        self.encoder1 = LinearNodeEncoder(dim_in, dim_emb)
        self.encoder2 = EquivStableLapPENodeEncoder(dim_emb, pecfg)

    def forward(self, batch):
        batch = self.encoder1(batch)
        batch = self.encoder2(batch)
        return batch


def build_node_encoder(cfg, dim_emb):
    """Return the node encoder matching the enabled positional encodings."""
    if cfg.posenc_EquivStableLapPE.enable:
        return EquivStableConcatNodeEncoder(cfg.dim_in, dim_emb,
                                            cfg.posenc_EquivStableLapPE)

    enabled = [name for name in pe_encs
               if getattr(cfg, f'posenc_{name}').enable]
    pecfgs = [getattr(cfg, f'posenc_{name}') for name in enabled]
    if len(enabled) == 0:
        return LinearNodeEncoder(cfg.dim_in, dim_emb)
    if len(enabled) == 1:
        return Concat2NodeEncoder(cfg.dim_in, dim_emb, enabled[0], pecfgs[0])
    return Concat3NodeEncoder(cfg.dim_in, dim_emb, enabled, pecfgs)
