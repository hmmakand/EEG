"""Model configuration for the GRIT network.

Replaces the GraphGym global `cfg` object used in the original repo
(`grit/config/gt_config.py`, `posenc_config.py`, GraphGym defaults).
Only fields that affect the *architecture* are kept; names follow the
original cfg keys:

    original cfg key                 ->  GritConfig field
    cfg.share.dim_in                 ->  dim_in
    (dataset task output size)       ->  dim_out
    cfg.gt.layers / n_heads          ->  layers / n_heads
    cfg.gt.dim_hidden/gnn.dim_inner  ->  dim_hidden
    cfg.gt.dropout / attn_dropout    ->  dropout / attn_dropout
    cfg.gt.layer_norm / batch_norm   ->  layer_norm / batch_norm
    cfg.gt.bn_momentum / bn_no_runner->  bn_momentum / bn_no_runner
    cfg.gt.update_e / rezero         ->  update_e / rezero
    cfg.gt.attn.*                    ->  attn.*
    cfg.gnn.act                      ->  act
    cfg.gnn.layers_pre_mp            ->  layers_pre_mp
    cfg.gnn.layers_post_mp           ->  layers_post_mp
    cfg.gnn.head                     ->  head
    cfg.model.graph_pooling          ->  graph_pooling
    cfg.dataset.node_encoder_name    ->  "LinearNode"
    cfg.dataset.edge_encoder_name    ->  "LinearEdge" (enabled if edge_dim)
    cfg.dataset.*_encoder_bn         ->  node_encoder_bn / edge_encoder_bn
    cfg.posenc_RRWP.*                ->  posenc_RRWP.*
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttnConfig:
    """Mirror of `cfg.gt.attn` (GRIT attention options)."""
    deg_scaler: bool = True      # degree-scaler after attention
    use_bias: bool = False       # bias in K and V projections
    clamp: Optional[float] = 5.  # clamp attention logits to [-clamp, clamp]
    act: str = 'relu'            # activation on the edge-conditioned score
    full_attn: bool = True       # pad the graph to fully-connected (all pairs)
    norm_e: bool = True          # normalise edge stream too
    O_e: bool = True             # output projection for the edge stream
    edge_enhance: bool = True    # add edge-score-weighted term to node output


@dataclass
class RRWPConfig:
    """Mirror of `cfg.posenc_RRWP` (see original `posenc_config.py`)."""
    enable: bool = True
    ksteps: int = 21             # K random-walk steps (= RRWP feature dim)


@dataclass
class GritConfig:
    # ---- input / output -------------------------------------------------
    dim_in: int = 1                 # node feature dim (batch.x.size(-1))
    dim_out: int = 1                # prediction dim (n classes / targets)
    edge_dim: Optional[int] = None  # edge feature dim; None -> no edge encoder

    # ---- GRIT backbone (cfg.gt.*) ----------------------------------------
    layers: int = 10
    n_heads: int = 8
    dim_hidden: int = 64
    dropout: float = 0.0
    attn_dropout: float = 0.2
    layer_norm: bool = False
    batch_norm: bool = True
    bn_momentum: float = 0.1
    bn_no_runner: bool = False       # BatchNorm without running stats
    update_e: bool = True            # pass updated edge features to next layer
    rezero: bool = False             # ReZero-style scaled residuals
    attn: AttnConfig = field(default_factory=AttnConfig)

    # ---- misc GNN (cfg.gnn.*) --------------------------------------------
    act: str = 'relu'
    layers_pre_mp: int = 0
    layers_post_mp: int = 2        # used by 'inductive_node' head
    head: str = 'san_graph'        # 'san_graph' | 'inductive_node'
    graph_pooling: str = 'add'     # 'mean' | 'add' | 'max'

    # ---- feature encoders (cfg.dataset.*) --------------------------------
    node_encoder_bn: bool = False  # the GRIT configs set these to False
    edge_encoder_bn: bool = False

    # ---- relative random-walk PE (cfg.posenc_RRWP) -----------------------
    posenc_RRWP: RRWPConfig = field(default_factory=RRWPConfig)

    @property
    def dim_edge(self) -> int:
        # Edge stream lives in the same space as the node stream.
        return self.dim_hidden
