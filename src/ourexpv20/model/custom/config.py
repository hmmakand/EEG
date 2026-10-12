"""Model configuration for the custom GPS + GRIT hybrid network.

Field names follow the GraphGPS and GRIT reference configurations where the
meaning is the same.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GritGPSConfig:
    # ---- input / output -------------------------------------------------
    dim_in: int = 8                 # node feature dim (x.size(-1))
    dim_out: int = 2                # prediction dim (n classes / targets)
    edge_dim: Optional[int] = None  # edge feature dim; None -> edge_attr ignored
    # Columns of x that are used; None -> all of them.
    feature_index: Optional[Tuple[int, ...]] = None
    # 'log_zscore': log(x), then standardise every feature across the nodes of
    #               its own graph. Removes the per-subject / per-trial power
    #               scale and keeps the spatial pattern over the electrodes.
    # 'log'       : log(x) only, for inputs already on a common scale
    #               (per-subject alignment).
    # 'none'      : use x as given.
    input_norm: str = 'log'
    # 'zscore': standardise every edge feature across the edges of its own
    #           graph before the edge encoder. 'none': use edge_attr as given.
    edge_norm: str = 'none'

    # ---- backbone ---------------------------------------------------------
    layers: int = 2
    n_heads: int = 4
    dim_hidden: int = 32            # node stream and pair stream share this size
    dropout: float = 0.3
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

    # ---- temporal encoder ---------------------------------------------------
    # Reads each node's time signal [N, samples] (forward argument `signal`).
    #   temporal_filters learned band-pass filters, shared by all electrodes
    #   -> square -> average pooling -> log   (time-resolved log power)
    # 'node' : the result is projected and added to the node embeddings.
    # 'head' : learned spatial filters over the electrodes are applied to the
    #          filter outputs before the power, and a linear layer adds the
    #          result to the logits (needs num_electrodes).
    # 'both' : both paths. 'none': the signal is not used.
    temporal: str = 'none'
    temporal_filters: int = 8
    temporal_kernel: int = 25       # samples (0.1 s at 250 Hz)
    temporal_pool: int = 75         # samples averaged per power estimate
    temporal_stride: int = 15
    temporal_spatial_filters: int = 16   # 'head' path
    temporal_dropout: float = 0.7
    signal_samples: int = 500       # length of `signal`

    # ---- head -------------------------------------------------------------
    graph_pooling: str = 'mean'     # 'mean' | 'add' | 'max'
    edge_readout: bool = True       # also pool the pair stream into the head
    head_dropout: float = 0.0       # dropout on the pooled graph embedding
    # Electrode-aware linear path added to the logits: one weight per
    # (electrode, node feature) and per (electrode pair, edge feature) of the
    # normalised inputs. Pooling averages over electrodes and pairs, so the
    # pooled head alone cannot weight a specific electrode pair. Needs
    # num_electrodes and graphs that all list those electrodes in order.
    wide_head: bool = False
    wide_dropout: float = 0.5
