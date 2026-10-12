"""Custom GPS + GRIT hybrid — the default classifier of this experiment.
Self-contained: it imports only torch and torch_geometric.

Takes the GPS layer layout (local MPNN branch + global attention branch + FFN)
and replaces GPS's structure-blind `MultiheadAttention` with GRIT's
pair-conditioned attention. Both branches read and write one shared pair
stream, and the head pools that stream together with the nodes.

File map:

    config.py    GritGPSConfig
    layer.py     GatedGCNLocal, GritAttention, GritGPSLayer
    model.py     GritGPSModel (encoders + on-the-fly RRWP + layers + head)

Minimal use (this experiment's graphs: 22 electrodes, 8 band powers,
thresholded PLV edges, no edge features), as built by `train/setup.py`:

    from model.custom import GritGPSConfig, GritGPSModel
    model = GritGPSModel(GritGPSConfig(dim_in=8, dim_out=2, num_electrodes=22))
    out = model(data.x, data.edge_index, data.batch)

With `include_edge_weights=True` in dataset/config.py, set `edge_dim=1` and
pass `edge_attr=data.edge_weight`.
"""
from .config import GritGPSConfig
from .layer import GatedGCNLocal, GritAttention, GritGPSLayer
from .model import GritGPSModel
