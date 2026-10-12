"""Model configuration for the custom GPS + GRIT hybrid network.

Field names follow `graphgps_net.GPSConfig` / `grit_net.GritConfig` where the
meaning is the same, so configs can be compared side by side.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class GritGPSConfig:
    # ---- input / output -------------------------------------------------
    dim_in: int = 8                 # node feature dim (x.size(-1))
    dim_out: int = 2                # prediction dim (n classes / targets)
    edge_dim: Optional[int] = None  # edge feature dim; None -> edge_attr ignored

    # ---- backbone ---------------------------------------------------------
    layers: int = 3
    n_heads: int = 4
    dim_hidden: int = 64            # node stream and pair stream share this size
    dropout: float = 0.0
    attn_dropout: float = 0.2
    layer_norm: bool = False
    batch_norm: bool = True
    act: str = 'relu'
    use_local: bool = True          # GatedGCN branch over the real edges
    use_global: bool = True         # GRIT attention branch over all node pairs

    # ---- GRIT attention options (cfg.gt.attn.* in GRIT) -------------------
    attn_clamp: Optional[float] = 5.  # clamp attention logits to [-clamp, clamp]
    attn_use_bias: bool = False       # bias in K and V projections
    edge_enhance: bool = True         # add pair-score-weighted term to node output

    # ---- positional information ------------------------------------------
    # Learnable embedding per node position inside a graph. Only valid when
    # every graph lists the same electrodes in the same order (it deliberately
    # breaks permutation invariance). None -> disabled.
    num_electrodes: Optional[int] = None
    # Relative random-walk probabilities [I, M, ..., M^(K-1)], computed on the
    # fly from the input graph. 0 -> disabled.
    rrwp_ksteps: int = 8
    # 'weighted': M is built from mean(|edge_attr|); falls back to 'binary'
    #             when no edge features are used (edge_dim=None).
    # 'binary'  : M = D^-1 A as in GRIT, for thresholded graphs.
    rrwp_weight: str = 'weighted'

    # ---- head -------------------------------------------------------------
    graph_pooling: str = 'mean'     # 'mean' | 'add' | 'max'
    edge_readout: bool = True       # also pool the pair stream into the head
