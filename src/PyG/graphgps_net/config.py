"""Model configuration for the GraphGPS network.

Replaces the GraphGym global `cfg` object used in the original repo
(`graphgps/config/gt_config.py`, `posenc_config.py`, GraphGym defaults).
Only the fields that affect the *architecture* are kept. Field names follow
the original cfg keys so configs can be ported one-to-one:

    original cfg key                 ->  GPSConfig field
    cfg.share.dim_in                 ->  dim_in
    (dataset task output size)       ->  dim_out
    cfg.gt.layer_type                ->  layer_type
    cfg.gt.layers                    ->  layers
    cfg.gt.n_heads                   ->  n_heads
    cfg.gt.dim_hidden/gnn.dim_inner  ->  dim_hidden
    cfg.gt.dropout                   ->  dropout
    cfg.gt.attn_dropout              ->  attn_dropout
    cfg.gt.layer_norm / batch_norm   ->  layer_norm / batch_norm
    cfg.gt.pna_degrees               ->  pna_degrees
    cfg.gnn.act                      ->  act
    cfg.gnn.layers_pre_mp            ->  layers_pre_mp
    cfg.gnn.layers_post_mp           ->  layers_post_mp
    cfg.gnn.head                     ->  head
    cfg.model.graph_pooling          ->  graph_pooling
    cfg.dataset.node_encoder_name    ->  "LinearNode" + pos_enc
    cfg.dataset.edge_encoder_name    ->  "LinearEdge" (enabled if edge_dim)
    cfg.dataset.*_encoder_bn         ->  node_encoder_bn / edge_encoder_bn
    cfg.posenc_<Name>.*              ->  posenc_<Name>.*
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PosEncConfig:
    """Mirror of `cfg.posenc_<Name>` (see original `posenc_config.py`)."""
    enable: bool = False
    dim_pe: int = 16              # size of the PE appended to node features
    model: str = 'DeepSet'        # LapPE: 'Transformer'|'DeepSet'; RWSE: 'mlp'|'linear'
    layers: int = 2               # layers in the PE encoder
    n_heads: int = 4              # heads if model == 'Transformer' (LapPE)
    post_layers: int = 0          # MLP layers after pooling (LapPE)
    raw_norm_type: str = 'none'   # 'none' | 'BatchNorm'
    pass_as_var: bool = False     # also keep PE as batch.pe_<Name>
    max_freqs: int = 10           # LapPE / EquivStableLapPE: num eigenvectors
    kernel_times: Tuple[int, ...] = tuple(range(1, 21))  # RWSE: walk steps


@dataclass
class GPSConfig:
    # ---- input / output -------------------------------------------------
    dim_in: int = 1                 # node feature dim (batch.x.size(-1))
    dim_out: int = 1                # prediction dim (n classes / targets)
    edge_dim: Optional[int] = None  # edge feature dim; None -> no edge encoder

    # ---- GPS backbone (cfg.gt.*) -----------------------------------------
    layer_type: str = 'CustomGatedGCN+Transformer'  # '<local MPNN>+<global attn>'
    layers: int = 3
    n_heads: int = 4
    dim_hidden: int = 64
    dropout: float = 0.0
    attn_dropout: float = 0.0
    layer_norm: bool = False
    batch_norm: bool = True
    pna_degrees: Optional[List[int]] = None   # required only for 'PNA'
    log_attn_weights: bool = False            # store attn maps (Transformer only)

    # ---- misc GNN (cfg.gnn.*) --------------------------------------------
    act: str = 'relu'
    layers_pre_mp: int = 0
    layers_post_mp: int = 2        # used by 'inductive_node' head
    head: str = 'san_graph'        # 'san_graph' | 'inductive_node'
    graph_pooling: str = 'mean'    # 'mean' | 'add' | 'max'

    # ---- feature encoders (cfg.dataset.*) --------------------------------
    node_encoder_bn: bool = True
    edge_encoder_bn: bool = True

    # ---- positional / structural encodings (cfg.posenc_*) ----------------
    posenc_LapPE: PosEncConfig = field(default_factory=PosEncConfig)
    posenc_RWSE: PosEncConfig = field(
        default_factory=lambda: PosEncConfig(model='linear'))
    posenc_EquivStableLapPE: PosEncConfig = field(default_factory=PosEncConfig)

    @property
    def local_gnn_type(self) -> str:
        return self.layer_type.split('+')[0]

    @property
    def global_model_type(self) -> str:
        return self.layer_type.split('+')[1]

    @property
    def dim_edge(self) -> int:
        # Original: hard-limit max edge dim for PNA (gps_model.FeatureEncoder).
        if 'PNA' in self.layer_type:
            return min(128, self.dim_hidden)
        return self.dim_hidden
