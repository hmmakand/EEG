"""Custom GPS + GRIT hybrid — standalone, imports nothing from `graphgps_net`
or `grit_net`.

Takes the GPS layer layout (local MPNN branch + global attention branch + FFN)
and replaces GPS's structure-blind `MultiheadAttention` with GRIT's
pair-conditioned attention. Both branches read and write one shared pair
stream, and the head pools that stream together with the nodes.

File map:

    config.py    GritGPSConfig
    layer.py     GatedGCNLocal, GritAttention, GritGPSLayer
    model.py     GritGPSModel (encoders + on-the-fly RRWP + layers + head)

Minimal use (version-seventeen graphs: 22 electrodes, 8 band powers,
thresholded PLV edges, no edge features):

    from custom import GritGPSConfig, GritGPSModel
    model = GritGPSModel(GritGPSConfig(dim_in=8, dim_out=2, num_electrodes=22))
    out = model(data.x, data.edge_index, data.batch)

With v17 `include_edge_weights=True`, set `edge_dim=1` and pass
`edge_attr=data.edge_weight`.
"""
from .config import GritGPSConfig
from .layer import GatedGCNLocal, GritAttention, GritGPSLayer
from .model import GritGPSModel
