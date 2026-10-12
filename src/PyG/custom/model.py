import torch
import torch.nn as nn
from torch_geometric.nn import (global_add_pool, global_max_pool,
                                global_mean_pool)
from torch_geometric.utils import coalesce, remove_self_loops

from .config import GritGPSConfig
from .layer import GritGPSLayer, act_dict

pooling_dict = {'add': global_add_pool,
                'mean': global_mean_pool,
                'max': global_max_pool}


class GritGPSModel(nn.Module):
    """GPS layer layout with GRIT attention as the global branch.

    Pipeline:
        node encoder   x -> Linear (+ electrode embedding) (+ Linear(RRWP_ii))
        pair stream    one vector per ordered node pair of each graph:
                       Linear(edge_attr) on the real edges, zero elsewhere,
                       (+ Linear(RRWP_ij))
        layers         GritGPSLayer x cfg.layers
        head           pool(nodes) || pool(pairs) -> MLP

    Called like the version-seventeen GAT, so the v17 training engine works
    unchanged:  out = model(data.x, data.edge_index, data.batch)

    Input : x [N, dim_in], edge_index [2, E], batch [N] (None -> one graph),
            edge_attr [E] or [E, edge_dim]       (only if cfg.edge_dim is set,
                                                  e.g. v17 `data.edge_weight`)
            Self-loops and duplicate edges are dropped first, which removes
            the (0, 0) placeholder edges of v17's legacy graph mode.
            Nothing has to be precomputed.
    Output: logits [num_graphs, dim_out]
    """

    def __init__(self, cfg: GritGPSConfig):
        super().__init__()
        self.cfg = cfg
        dim_h = cfg.dim_hidden

        self.node_encoder = nn.Linear(cfg.dim_in, dim_h)
        if cfg.num_electrodes is not None:
            self.electrode_emb = nn.Embedding(cfg.num_electrodes, dim_h)
        if cfg.edge_dim is not None:
            self.edge_encoder = nn.Linear(cfg.edge_dim, dim_h)

        if cfg.rrwp_weight not in ('weighted', 'binary'):
            raise ValueError(f"Unexpected rrwp_weight: {cfg.rrwp_weight}")
        if cfg.rrwp_ksteps > 0:
            self.rrwp_abs_encoder = nn.Linear(cfg.rrwp_ksteps, dim_h,
                                              bias=False)
            self.rrwp_rel_encoder = nn.Linear(cfg.rrwp_ksteps, dim_h,
                                              bias=False)

        self.layers = nn.ModuleList([
            GritGPSLayer(
                dim_h=dim_h,
                num_heads=cfg.n_heads,
                act=cfg.act,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                layer_norm=cfg.layer_norm,
                batch_norm=cfg.batch_norm,
                use_local=cfg.use_local,
                use_global=cfg.use_global,
                attn_clamp=cfg.attn_clamp,
                attn_use_bias=cfg.attn_use_bias,
                edge_enhance=cfg.edge_enhance,
            ) for _ in range(cfg.layers)])

        self.pooling_fun = pooling_dict[cfg.graph_pooling]
        dim_graph = dim_h * 2 if cfg.edge_readout else dim_h
        self.post_mp = nn.Sequential(
            nn.Linear(dim_graph, dim_h),
            act_dict[cfg.act](),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim_h, cfg.dim_out))

    @torch.no_grad()
    def _rrwp(self, weight, b_src, l_src, l_dst, num_graphs, n_max):
        """[I, M, ..., M^(K-1)] per graph, shape [B, n_max, n_max, K]."""
        A = weight.new_zeros(num_graphs, n_max, n_max)
        A.index_put_((b_src, l_src, l_dst), weight, accumulate=True)
        M = A / A.sum(-1, keepdim=True).clamp(min=1e-12)
        P = torch.eye(n_max, device=A.device).expand_as(A)
        powers = [P]
        for _ in range(self.cfg.rrwp_ksteps - 1):
            P = P @ M
            powers.append(P)
        return torch.stack(powers, dim=-1)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        cfg = self.cfg
        num_nodes = x.size(0)
        # A single un-batched graph has no `batch` vector; treat it as graph 0.
        bvec = batch
        if bvec is None:
            bvec = x.new_zeros(num_nodes, dtype=torch.long)

        if cfg.edge_dim is None:
            edge_attr = None
        elif edge_attr is None:
            raise ValueError(f"edge_attr is required when edge_dim="
                             f"{cfg.edge_dim}")
        else:
            edge_attr = edge_attr.view(-1, cfg.edge_dim)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        if edge_attr is None:
            edge_index = coalesce(edge_index, num_nodes=num_nodes)
        else:
            edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                             reduce='mean')

        # Position of every node inside its own graph.
        counts = torch.bincount(bvec)
        num_graphs, n_max = counts.numel(), int(counts.max())
        ptr = counts.cumsum(0) - counts
        local = torch.arange(num_nodes, device=x.device) - ptr[bvec]

        # All ordered node pairs (self pairs included) of every graph.
        mask = torch.arange(n_max, device=x.device)[None] < counts[:, None]
        mask2d = mask[:, :, None] & mask[:, None, :]
        b, i, j = mask2d.nonzero(as_tuple=True)
        pair_index = torch.stack([ptr[b] + i, ptr[b] + j])
        pair_batch = b
        num_pairs = b.numel()

        # Row of every real edge in the pair stream.
        src, dst = edge_index
        pair_id = torch.full(mask2d.shape, -1, dtype=torch.long,
                             device=x.device)
        pair_id[mask2d] = torch.arange(num_pairs, device=x.device)
        edge_pos = pair_id[bvec[src], local[src], local[dst]]

        # Node and pair encoders.
        h = self.node_encoder(x)
        if cfg.num_electrodes is not None:
            h = h + self.electrode_emb(local)
        e = h.new_zeros(num_pairs, cfg.dim_hidden)
        if edge_attr is not None:
            e = e.index_add(0, edge_pos, self.edge_encoder(edge_attr))

        if cfg.rrwp_ksteps > 0:
            if cfg.rrwp_weight == 'weighted' and edge_attr is not None:
                weight = edge_attr.abs().mean(-1)
            else:
                weight = h.new_ones(edge_index.size(1))
            rrwp = self._rrwp(weight, bvec[src], local[src], local[dst],
                              num_graphs, n_max)
            rrwp_abs = rrwp.diagonal(dim1=1, dim2=2).permute(0, 2, 1)[mask]
            h = h + self.rrwp_abs_encoder(rrwp_abs)
            e = e + self.rrwp_rel_encoder(rrwp[mask2d])

        for layer in self.layers:
            h, e = layer(h, e, edge_index, edge_pos, pair_index)

        graph_emb = self.pooling_fun(h, bvec, size=num_graphs)
        if cfg.edge_readout:
            graph_emb = torch.cat(
                [graph_emb, self.pooling_fun(e, pair_batch, size=num_graphs)],
                dim=-1)
        return self.post_mp(graph_emb)
