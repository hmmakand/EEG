"""GRIT network only — standalone extraction of the model from
https://github.com/LiamMa/GRIT (no GraphGym cfg/registry, no training,
no data loading, no preprocessing transforms).

File map (mirrors the original `grit/` layout):

    config.py                    GritConfig / AttnConfig / RRWPConfig (replaces cfg)
    act.py                       activation registry
    encoder/linear_node_encoder.py   x         -> Linear
    encoder/linear_edge_encoder.py   edge_attr -> Linear
    encoder/rrwp_encoder.py          RRWP node + pair encoders, full-graph padding
    layer/grit_layer.py              GRIT sparse attention + Transformer layer
    head/san_graph.py                graph-level head (pool + MLP)
    head/inductive_node.py           node-level head
    network/grit_model.py            FeatureEncoder + GritTransformer

Minimal use:

    from grit_net import GritConfig, GritTransformer
    model = GritTransformer(GritConfig(dim_in=5, edge_dim=31, dim_out=2))
    pred, _ = model(batch)   # batch needs rrwp, rrwp_index, rrwp_val, log_deg
"""
from .config import AttnConfig, GritConfig, RRWPConfig
from .layer import GritTransformerLayer
from .network import GritTransformer
