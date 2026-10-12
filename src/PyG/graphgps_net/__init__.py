"""GraphGPS network only — standalone extraction of the model from
https://github.com/rampasek/GraphGPS (no GraphGym cfg/registry, no training,
no data loading).

File map (mirrors the original `graphgps/` layout):

    config.py                              GPSConfig / PosEncConfig  (replaces cfg)
    act.py                                 activation registry
    encoder/linear_node_encoder.py         x         -> Linear
    encoder/linear_edge_encoder.py         edge_attr -> Linear
    encoder/laplace_pos_encoder.py         LapPE  (DeepSet / Transformer)
    encoder/kernel_pos_encoder.py          RWSE   (mlp / linear)
    encoder/equivstable_laplace_pos_encoder.py
    encoder/composed_encoders.py           node encoder + PE concatenation
    layer/gatedgcn_layer.py                GatedGCN local MPNN
    layer/gine_conv_layer.py               GINE with EquivStableLapPE
    layer/gps_layer.py                     GPS layer: local MPNN + global attn + FFN
    head/san_graph.py                      graph-level head (pool + MLP)
    head/inductive_node.py                 node-level head
    network/gps_model.py                   FeatureEncoder + GPSModel

Minimal use:

    from graphgps_net import GPSConfig, GPSModel
    model = GPSModel(GPSConfig(dim_in=5, edge_dim=31, dim_out=2))
    pred, _ = model(batch)
"""
from .config import GPSConfig, PosEncConfig
from .layer import GPSLayer
from .network import GPSModel
