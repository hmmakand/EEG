import torch
import torch.nn as nn

from ..act import act_dict
from ..config import GritConfig
from ..encoder import (LinearEdgeEncoder, LinearNodeEncoder,
                       RRWPLinearEdgeEncoder, RRWPLinearNodeEncoder)
from ..head import head_dict
from ..layer.grit_layer import GritTransformerLayer


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

        x         [N, dim_in]   -> LinearNodeEncoder -> [N, dim_hidden]
        edge_attr [E, edge_dim] -> LinearEdgeEncoder -> [E, dim_edge]

    Args:
        cfg (GritConfig): model configuration
    """
    def __init__(self, cfg: GritConfig):
        super().__init__()
        self.dim_in = cfg.dim_in
        self.node_encoder = LinearNodeEncoder(cfg.dim_in, cfg.dim_hidden)
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


class GritTransformer(nn.Module):
    '''
        The proposed GritTransformer (Graph Inductive Bias Transformer)
        Ma, L., Lin, C., Lim, D., Romero-Soriano, A., Dokania, P. K., Coates,
        M., Torr, P., & Lim, S.-N. Graph Inductive Biases in Transformers
        without Message Passing. (ICML 2023) https://arxiv.org/abs/2305.17589

    Pipeline:
        encoder            x, edge_attr -> dim_hidden
        rrwp_abs_encoder   x += Linear(rrwp)                     (node PE)
        rrwp_rel_encoder   edge_index/edge_attr -> all node pairs,
                           edge_attr += Linear(rrwp_val)         (pair PE)
        [pre_mp]
        layers             GritTransformerLayer x cfg.layers
        post_mp            head

    Input : a torch_geometric `Data`/`Batch` with
              x [N, dim_in], edge_index [2, E], batch [N]
              edge_attr [E, edge_dim]                 (if cfg.edge_dim is set)
              rrwp [N, K], rrwp_index [2, P], rrwp_val [P, K]   (if RRWP)
              log_deg [N] (or deg [N])  -- degree of the ORIGINAL graph,
                                           used by the degree scaler
    Output: (pred, label) — pred is [num_graphs, dim_out] for 'san_graph' or
            [N, dim_out] for 'inductive_node'; label is batch.y or None.
    '''

    def __init__(self, cfg: GritConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = FeatureEncoder(cfg)
        dim_in = self.encoder.dim_in

        if cfg.posenc_RRWP.enable:
            self.rrwp_abs_encoder = RRWPLinearNodeEncoder(
                cfg.posenc_RRWP.ksteps, cfg.dim_hidden)
            rel_pe_dim = cfg.posenc_RRWP.ksteps
            self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(
                rel_pe_dim, cfg.dim_edge,
                pad_to_full_graph=cfg.attn.full_attn,
                add_node_attr_as_self_loop=False,
                fill_value=0.)

        if cfg.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.dim_hidden, cfg.layers_pre_mp,
                                   act=cfg.act)
            dim_in = cfg.dim_hidden

        assert cfg.dim_hidden == dim_in, \
            "The inner and hidden dims must match."

        layers = []
        for _ in range(cfg.layers):
            layers.append(GritTransformerLayer(
                in_dim=cfg.dim_hidden,
                out_dim=cfg.dim_hidden,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                act=cfg.act,
                attn_dropout=cfg.attn_dropout,
                layer_norm=cfg.layer_norm,
                batch_norm=cfg.batch_norm,
                residual=True,
                norm_e=cfg.attn.norm_e,
                O_e=cfg.attn.O_e,
                attn_cfg=cfg.attn,
                bn_momentum=cfg.bn_momentum,
                bn_no_runner=cfg.bn_no_runner,
                update_e=cfg.update_e,
                rezero=cfg.rezero,
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
