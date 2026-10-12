"""GRIT attention + Transformer layer.

Original: `grit/layer/grit_layer.py`
Changes: torch_scatter -> torch_geometric.utils.scatter, opt_einsum ->
torch.einsum, GraphGym cfg -> explicit arguments. Unused constructor flags
(sqrt_relu, signed_sqrt, scaled_attn, no_qk) and the broken 'graphormer_attn'
branch (class was never imported) are removed.
"""
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

from ..act import act_dict
from ..config import AttnConfig


def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter(src, index, dim=0, dim_size=num_nodes,
                        reduce='max')[index]
    out = out.exp()
    out = out / (scatter(out, index, dim=0, dim_size=num_nodes,
                         reduce='sum')[index] + 1e-16)
    return out


class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
        Proposed Attention Computation for GRIT

        For every (src j -> dst i) pair in edge_index, per head:
            s_ij   = K_j + Q_i                               (additive)
            s_ij   = signed_sqrt(s_ij * E_w) + E_b           (edge-conditioned)
            e_ij   = act(s_ij)                               -> new edge feat
            a_ij   = softmax_j( clamp(Aw . e_ij) )
            h_i    = sum_j a_ij * V_j  (+ VeRow . sum_j a_ij * e_ij)
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1),
                               requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(
                torch.zeros(self.out_dim, self.num_heads, self.out_dim),
                requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        num_nodes = batch.num_nodes
        src = batch.K_h[batch.edge_index[0]]   # (num relative) x heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num relative) x heads x out_dim
        score = src + dest

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
            # (num relative) x num_heads x out_dim
            score = score * E_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + E_b

        score = self.act(score)
        e_t = score

        # output edge
        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        # final attn
        score = torch.einsum("ehd,dhc->ehc", score, self.Aw)
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, batch.edge_index[1], num_nodes)  # E x heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # E x heads x out_dim
        batch.wV = scatter(msg, batch.edge_index[1], dim=0,
                           dim_size=num_nodes, reduce='sum')

        if self.edge_enhance and batch.E is not None:
            # NOTE: original omitted dim_size here; added so nodes without
            # incoming pairs (sparse attention) cannot cause a size mismatch.
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0,
                           dim_size=num_nodes, reduce='sum')
            rowV = torch.einsum("nhd,dhc->nhc", rowV, self.VeRow)
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out


class GritTransformerLayer(nn.Module):
    """
        Proposed Transformer Layer for GRIT

        h, e = GritAttention(h, e)
        h = Norm(h_in + O_h(deg_scale(h)))      e = Norm(e_in + O_e(e))
        h = Norm(h + FFN(h))                     FFN: d -> 2d -> d
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 attn_cfg: AttnConfig = AttnConfig(),
                 bn_momentum=0.1,
                 bn_no_runner=False,
                 update_e=True,
                 rezero=False,
                 **kwargs):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # -------
        self.update_e = update_e
        self.bn_momentum = bn_momentum
        self.bn_no_runner = bn_no_runner
        self.rezero = rezero

        self.act = act_dict[act]() if act is not None else nn.Identity()
        self.deg_scaler = attn_cfg.deg_scaler

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=attn_cfg.use_bias,
            dropout=attn_dropout,
            clamp=attn_cfg.clamp,
            act=attn_cfg.act,
            edge_enhance=attn_cfg.edge_enhance,
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()

        # -------- Deg Scaler Option ------
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(
                torch.zeros(1, out_dim // num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to
            # avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(
                out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                momentum=bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(
                out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                momentum=bn_momentum) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(
                out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                momentum=bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h  # for first residual connection
        e_in1 = batch.get("edge_attr", None)
        e = None
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero:
                h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual,
            super().__repr__(),
        )


@torch.no_grad()
def get_log_deg(batch):
    """log(deg + 1) per node, shape [N, 1].

    Prefers `batch.log_deg` / `batch.deg` precomputed on the ORIGINAL graph.
    Falling back to edge_index is wrong once the RRWP encoder has padded the
    graph to fully-connected (every node then has degree N).
    """
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        warnings.warn("Compute the degree on the fly; Might be problematic if "
                      "have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.edge_index[1],
                               num_nodes=batch.num_nodes,
                               dtype=torch.float)
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg
