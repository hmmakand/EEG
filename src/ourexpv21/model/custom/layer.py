"""Layers of the custom GPS + GRIT hybrid.

    GatedGCNLocal   local branch  (GatedGCN as used by GraphGPS)
    GritAttention   global branch (GRIT sparse attention)
    GritGPSLayer    GPS-style layer that runs both on one shared pair stream

Every module takes and returns plain tensors.
"""
import torch
import torch.nn as nn
from torch_geometric.utils import scatter, softmax

act_dict = {'relu': nn.ReLU, 'gelu': nn.GELU, 'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU, 'identity': nn.Identity}


def make_norm(dim, layer_norm, batch_norm):
    if layer_norm and batch_norm:
        raise ValueError("Cannot apply two types of normalization together")
    if layer_norm:
        return nn.LayerNorm(dim)
    if batch_norm:
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class GatedGCNLocal(nn.Module):
    """
        GatedGCN message passing over the real edges (src j -> dst i):
            e_ij = D h_i + E h_j + C e_ij
            h_i  = A h_i + sum_j sigmoid(e_ij) * B h_j / sum_j sigmoid(e_ij)

    Returns the node and edge *updates*; the residual connection is applied
    by the enclosing layer.
    """

    def __init__(self, dim, dropout, act='relu'):
        super().__init__()
        self.A = nn.Linear(dim, dim)
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)
        self.D = nn.Linear(dim, dim)
        self.E = nn.Linear(dim, dim)
        self.bn_node_x = nn.BatchNorm1d(dim)
        self.bn_edge_e = nn.BatchNorm1d(dim)
        self.act_fn_x = act_dict[act]()
        self.act_fn_e = act_dict[act]()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, e, edge_index):
        """
        x               : [n_nodes, dim]
        e               : [n_edges, dim]
        edge_index      : [2, n_edges]
        """
        src, dst = edge_index
        e_ij = self.D(x)[dst] + self.E(x)[src] + self.C(e)
        sigma_ij = torch.sigmoid(e_ij)

        num = scatter(sigma_ij * self.B(x)[src], dst, 0, dim_size=x.size(0),
                      reduce='sum')
        den = scatter(sigma_ij, dst, 0, dim_size=x.size(0), reduce='sum')
        x = self.A(x) + num / (den + 1e-6)

        x = self.dropout(self.act_fn_x(self.bn_node_x(x)))
        # A batch can hold no edge at all after thresholding; BatchNorm cannot
        # take training statistics from fewer than two rows.
        if e_ij.size(0) > 1 or not self.training:
            e_ij = self.bn_edge_e(e_ij)
        e = self.dropout(self.act_fn_e(e_ij))
        return x, e


class GritAttention(nn.Module):
    """
        GRIT attention over node pairs (src j -> dst i), per head:
            s_ij   = K_j + Q_i                               (additive)
            s_ij   = signed_sqrt(s_ij * E_w) + E_b           (pair-conditioned)
            e_ij   = act(s_ij)                               -> new pair feat
            a_ij   = softmax_j( clamp(Aw . e_ij) )
            h_i    = sum_j a_ij * V_j  (+ VeRow . sum_j a_ij * e_ij)
    """

    def __init__(self, dim, num_heads, use_bias=False, clamp=5., dropout=0.,
                 act='relu', edge_enhance=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim_hidden={dim} must be divisible by "
                             f"n_heads={num_heads}")
        self.num_heads = num_heads
        self.out_dim = dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(dim, dim, bias=True)
        self.K = nn.Linear(dim, dim, bias=use_bias)
        self.E = nn.Linear(dim, dim * 2, bias=True)
        self.V = nn.Linear(dim, dim, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1))
        nn.init.xavier_normal_(self.Aw)
        self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(
                torch.zeros(self.out_dim, self.num_heads, self.out_dim))
            nn.init.xavier_normal_(self.VeRow)

    def forward(self, x, e, pair_index):
        """
        x               : [n_nodes, dim]
        e               : [n_pairs, dim]
        pair_index      : [2, n_pairs]
        Returns h [n_nodes, dim], e [n_pairs, dim], attn [n_pairs, heads, 1].
        """
        num_nodes = x.size(0)
        src, dst = pair_index
        Q_h = self.Q(x).view(-1, self.num_heads, self.out_dim)
        K_h = self.K(x).view(-1, self.num_heads, self.out_dim)
        V_h = self.V(x).view(-1, self.num_heads, self.out_dim)
        E = self.E(e).view(-1, self.num_heads, self.out_dim * 2)
        E_w, E_b = E[:, :, :self.out_dim], E[:, :, self.out_dim:]

        score = (K_h[src] + Q_h[dst]) * E_w
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        score = self.act(score + E_b)
        e_t = score  # n_pairs x heads x out_dim

        score = torch.einsum("ehd,dhc->ehc", score, self.Aw)
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)
        attn = softmax(score, dst, num_nodes=num_nodes)  # n_pairs x heads x 1
        score = self.dropout(attn)

        h = scatter(V_h[src] * score, dst, dim=0, dim_size=num_nodes,
                    reduce='sum')
        if self.edge_enhance:
            rowV = scatter(e_t * score, dst, dim=0, dim_size=num_nodes,
                           reduce='sum')
            h = h + torch.einsum("nhd,dhc->nhc", rowV, self.VeRow)

        return h.flatten(1), e_t.flatten(1), attn


class TemporalEncoder(nn.Module):
    """Learned band-pass filters and log power over time for node signals.

        u = BN(Conv1d(signal))                          [N, F, T]  per electrode
        node path : log(avgpool(u^2))                   -> Linear -> [N, dim_h]
        head path : s = spatial filters over electrodes [B, S, T]
                    log(avgpool(BN(s)^2))               -> Linear -> [B, dim_out]
        edge path : c_ij = mean_t(u_i * u_j) per filter [B, N, N, F]
                    log(c_ii)                           -> Linear -> [N, dim_h]
                    BN(c_ij)                            -> Linear -> [B*N*N, dim_h]

    The edge path gives the graph model the covariance of every electrode pair
    in each learned band, as pair features that condition the attention; the
    attention then does the spatial filtering that the head path does directly.

    The head path is the ShallowConvNet idea (Schirrmeister et al., 2017): a
    spatial filter followed by log power is what CSP computes, with the filters
    learned jointly with the temporal ones.
    """

    def __init__(self, dim_h, dim_out, mode, num_electrodes, signal_samples,
                 filters=8, kernel=25, pool=75, stride=15, spatial_filters=16,
                 dropout=0.5, crop=0, crop_stride=25, edge_dropout=0.3):
        super().__init__()
        if mode not in ('node', 'head', 'both', 'edge', 'edge+head'):
            raise ValueError(f"Unexpected temporal mode: {mode}")
        self.node_path = mode in ('node', 'both')
        self.head_path = mode in ('head', 'both', 'edge+head')
        self.edge_path = mode in ('edge', 'edge+head')
        if (self.head_path or self.edge_path) and num_electrodes is None:
            raise ValueError("temporal head and edge paths require num_electrodes")
        self.num_electrodes = num_electrodes
        self.conv = nn.Conv1d(1, filters, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(filters)
        self.pool = nn.AvgPool1d(pool, stride)
        self.dropout = nn.Dropout(dropout)
        if crop < 0 or crop > signal_samples or crop_stride < 1:
            raise ValueError("temporal_crop must be within the signal and "
                             "temporal_crop_stride positive")
        self.crop = crop
        self.crop_stride = crop_stride
        conv_samples = (crop or signal_samples) + 2 * (kernel // 2) - kernel + 1
        steps = (conv_samples - pool) // stride + 1
        if steps < 1:
            raise ValueError("signal_samples is too short for temporal_pool")
        if self.node_path:
            self.node_out = nn.Linear(filters * steps, dim_h)
        if self.head_path:
            self.spatial = nn.Parameter(
                torch.empty(spatial_filters, filters, num_electrodes))
            nn.init.xavier_uniform_(self.spatial)
            self.bn_spatial = nn.BatchNorm1d(spatial_filters)
            self.head_out = nn.Linear(spatial_filters * steps, dim_out)
        if self.edge_path:
            self.bn_edge = nn.BatchNorm1d(filters)
            self.edge_dropout = nn.Dropout(edge_dropout)
            self.edge_node_out = nn.Linear(filters, dim_h)
            self.edge_out = nn.Linear(filters, dim_h)

    def _log_power(self, u):
        return torch.log(self.pool(u ** 2).clamp(min=1e-6))

    def forward(self, signal, num_graphs, batch=None):
        """signal [N, samples] -> (node embedding [N, dim_h] or None,
                                   logits [num_graphs, dim_out] or None,
                                   pair embedding [num_graphs * n * n, dim_h]
                                   or None, pairs ordered graph, row, column).

        With cropping, training uses one random window per graph (`batch`
        maps nodes to graphs) and evaluation averages over regular windows.
        """
        if not self.crop:
            return self._encode(signal, num_graphs)
        last = signal.size(1) - self.crop
        if self.training:
            offset = torch.randint(0, last + 1, (num_graphs,),
                                   device=signal.device)
            index = offset[batch].unsqueeze(1) + torch.arange(
                self.crop, device=signal.device)
            return self._encode(signal.gather(1, index), num_graphs)
        outputs = [self._encode(signal[:, start:start + self.crop], num_graphs)
                   for start in range(0, last + 1, self.crop_stride)]
        return tuple(None if parts[0] is None else torch.stack(parts).mean(0)
                     for parts in zip(*outputs))

    def _encode(self, signal, num_graphs):
        u = self.bn(self.conv(signal.unsqueeze(1)))  # [N, F, T]
        h = logits = pairs = None
        if self.node_path:
            h = self.node_out(self.dropout(self._log_power(u).flatten(1)))
        if self.head_path or self.edge_path:
            if signal.size(0) != num_graphs * self.num_electrodes:
                raise ValueError("temporal head and edge paths need "
                                 "num_electrodes nodes per graph")
            u = u.view(num_graphs, self.num_electrodes, *u.shape[1:])
        if self.edge_path:
            cov = torch.einsum('bift,bjft->bijf', u, u) / u.size(-1)
            power = cov.diagonal(dim1=1, dim2=2).permute(0, 2, 1)  # [B, n, F]
            h = self.edge_node_out(self.edge_dropout(
                torch.log(power.clamp(min=1e-6)).flatten(0, 1)))
            pairs = self.edge_out(self.edge_dropout(
                self.bn_edge(cov.flatten(0, 2))))
        if self.head_path:
            s = self.bn_spatial(torch.einsum('sfn,bnft->bst', self.spatial, u))
            logits = self.head_out(self.dropout(self._log_power(s).flatten(1)))
        return h, logits, pairs


class GritGPSLayer(nn.Module):
    """Local GatedGCN + GRIT pair-conditioned attention on a shared pair stream.

        h_loc, e_loc = GatedGCN(h, e[real edges])        # local branch
        h_att, e_att = GritAttention(h, e)               # global branch
        h = Norm(h + h_loc) + Norm(h + O_h(h_att))
        e = Norm(e + O_e(e_att) + e_loc on the real edges)
        h = Norm(h + FFN(h))                             # FFN: d -> 2d -> d

    `e` holds one vector for every ordered node pair of each graph (self pairs
    included); `edge_pos` gives the row of every real edge in that pair stream.
    """

    def __init__(self, dim_h, num_heads, act='relu', dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 use_local=True, use_global=True, attn_clamp=5.,
                 attn_use_bias=False, edge_enhance=True):
        super().__init__()
        if not (use_local or use_global):
            raise ValueError("At least one of use_local / use_global is needed")
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.use_local = use_local
        self.use_global = use_global
        self.attn_weights = None  # last attention map [n_pairs, heads, 1]

        if use_local:
            self.local_model = GatedGCNLocal(dim_h, dropout, act=act)
            self.norm1_local = make_norm(dim_h, layer_norm, batch_norm)

        if use_global:
            self.self_attn = GritAttention(
                dim_h, num_heads, use_bias=attn_use_bias, clamp=attn_clamp,
                dropout=attn_dropout, act=act, edge_enhance=edge_enhance)
            self.O_h = nn.Linear(dim_h, dim_h)
            self.O_e = nn.Linear(dim_h, dim_h)
            self.norm1_attn = make_norm(dim_h, layer_norm, batch_norm)
        self.norm1_e = make_norm(dim_h, layer_norm, batch_norm)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = act_dict[act]()
        self.norm2 = make_norm(dim_h, layer_norm, batch_norm)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, h, e, edge_index, edge_pos, pair_index):
        h_in1 = h  # for first residual connection
        e_in1 = e

        h_out_list = []
        # Local MPNN over the real edges.
        if self.use_local:
            h_local, e_local = self.local_model(h, e_in1[edge_pos], edge_index)
            h_out_list.append(self.norm1_local(h_in1 + h_local))
            e = e.index_add(0, edge_pos, e_local)

        # Pair-conditioned attention over all node pairs.
        if self.use_global:
            h_attn, e_attn, attn = self.self_attn(h, e_in1, pair_index)
            self.attn_weights = attn.detach()
            h_attn = self.dropout_attn(self.O_h(h_attn))
            h_out_list.append(self.norm1_attn(h_in1 + h_attn))
            e = e + self.dropout_attn(self.O_e(e_attn))

        # Combine local and global outputs.
        h = sum(h_out_list)
        e = self.norm1_e(e)

        # Feed Forward block.
        h = self.norm2(h + self._ff_block(h))
        return h, e

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        return f'summary: dim_h={self.dim_h}, use_local={self.use_local}, ' \
               f'use_global={self.use_global}, heads={self.num_heads}'
