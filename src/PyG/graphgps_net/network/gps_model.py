import torch
import torch.nn as nn

from ..act import act_dict
from ..config import GPSConfig
from ..encoder import LinearEdgeEncoder, build_node_encoder
from ..head import head_dict
from ..layer.gps_layer import GPSLayer


class BatchNorm1dNode(nn.Module):
    """BatchNorm over node features `batch.x` (GraphGym BatchNorm1dNode)."""

    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(nn.Module):
    """BatchNorm over edge features `batch.edge_attr`.

    NOTE: the original repo wrapped the edge encoder output in GraphGym's
    `BatchNorm1dNode`, which normalises `batch.x` rather than `edge_attr`.
    This class applies the normalisation to the edges as intended.
    """

    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


class FeatureEncoder(nn.Module):
    """
    Encoding node and edge features

        x         [N, dim_in]   -> node encoder (+ PEs) -> [N, dim_hidden]
        edge_attr [E, edge_dim] -> LinearEdgeEncoder     -> [E, dim_edge]

    Args:
        cfg (GPSConfig): model configuration
    """
    def __init__(self, cfg: GPSConfig):
        super().__init__()
        self.dim_in = cfg.dim_in
        # Encode node features (+ positional encodings if enabled)
        self.node_encoder = build_node_encoder(cfg, cfg.dim_hidden)
        if cfg.node_encoder_bn:
            self.node_encoder_bn = BatchNorm1dNode(cfg.dim_hidden)
        # Update dim_in to reflect the new dimension of the node features
        self.dim_in = cfg.dim_hidden

        if cfg.edge_dim is not None:
            self.edge_encoder = LinearEdgeEncoder(cfg.edge_dim, cfg.dim_edge)
            if cfg.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.dim_edge)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GNNPreMP(nn.Module):
    """Linear pre-message-passing layers (GraphGym GNNPreMP):
    [Linear -> BatchNorm -> act] x num_layers, applied to batch.x."""

    def __init__(self, dim_in, dim_out, num_layers, act='relu'):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(dim_in if i == 0 else dim_out, dim_out,
                                 bias=False),
                       nn.BatchNorm1d(dim_out),
                       act_dict[act]()]
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        batch.x = self.model(batch.x)
        return batch


class GPSModel(nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)

    Pipeline:  encoder -> [pre_mp] -> GPSLayer x cfg.layers -> post_mp (head)

    Input : a torch_geometric `Data`/`Batch` with
              x [N, dim_in], edge_index [2, E], batch [N],
              edge_attr [E, edge_dim]      (if cfg.edge_dim is set)
              EigVals / EigVecs            (if LapPE / EquivStableLapPE)
              pestat_RWSE                  (if RWSE)
              attn_bias                    (if 'BiasedTransformer')
    Output: (pred, label) — pred is [num_graphs, dim_out] for 'san_graph' or
            [N, dim_out] for 'inductive_node'; label is batch.y or None.
    """

    def __init__(self, cfg: GPSConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = FeatureEncoder(cfg)
        dim_in = self.encoder.dim_in

        if cfg.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.dim_hidden, cfg.layers_pre_mp,
                                   act=cfg.act)
            dim_in = cfg.dim_hidden

        if not cfg.dim_hidden == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.dim_hidden} dim_in={dim_in}")

        try:
            local_gnn_type, global_model_type = cfg.layer_type.split('+')
        except ValueError:
            raise ValueError(f"Unexpected layer type: {cfg.layer_type}")
        layers = []
        for _ in range(cfg.layers):
            layers.append(GPSLayer(
                dim_h=cfg.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.n_heads,
                act=cfg.act,
                pna_degrees=cfg.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                layer_norm=cfg.layer_norm,
                batch_norm=cfg.batch_norm,
                log_attn_weights=cfg.log_attn_weights,
            ))
        self.layers = nn.Sequential(*layers)

        if cfg.head == 'san_graph':
            self.post_mp = head_dict['san_graph'](
                dim_in=cfg.dim_hidden, dim_out=cfg.dim_out,
                graph_pooling=cfg.graph_pooling, act=cfg.act)
        else:
            self.post_mp = head_dict[cfg.head](
                dim_in=cfg.dim_hidden, dim_out=cfg.dim_out,
                layers_post_mp=cfg.layers_post_mp, act=cfg.act)

    def forward(self, batch):
        # A single un-batched graph has no `batch` vector; treat it as graph 0.
        if getattr(batch, 'batch', None) is None:
            batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long,
                                      device=batch.x.device)
        for module in self.children():
            batch = module(batch)
        return batch
