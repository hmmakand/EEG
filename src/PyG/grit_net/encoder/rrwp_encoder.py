'''
    The RRWP encoder for GRIT

    RRWP = Relative Random Walk Probabilities. With M = D^-1 A and K steps:
        P_ij = [I, M, M^2, ..., M^(K-1)]_ij        (K-dim vector per node pair)
    - absolute part  P_ii  -> batch.rrwp       [N, K]          (node PE)
    - relative part  P_ij  -> batch.rrwp_index [2, n_pairs],
                              batch.rrwp_val   [n_pairs, K]    (edge PE)
    These are precomputed per graph by the data transform (`grit/transform/
    rrwp.py` in the original repo, not included here).

Original: `grit/encoder/rrwp_encoder.py`
(torch_sparse.coalesce -> torch_geometric.utils.coalesce, torch_scatter ->
torch_geometric.utils.scatter; the unused 'masked_rrwp_linear' and
'pad_to_full_graph' encoders are removed.)
'''
import warnings

import torch
from torch import nn
from torch_geometric.utils import add_self_loops, coalesce, scatter


def full_edge_index(edge_index, batch=None):
    """
    Return the edge index of the fully-connected version of every graph in
    the batch (all N_g x N_g ordered pairs, self-loops included).

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Full edge index.
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    full_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        adj = torch.ones((n, n), dtype=torch.short, device=edge_index.device)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        full_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(full_index_list, dim=1).contiguous()
    return edge_index_full


class RRWPLinearNodeEncoder(torch.nn.Module):
    """
        FC_1(RRWP) + FC_2 (Node-attr)
        note: FC_2 is the node encoder applied before (LinearNodeEncoder).

        batch.rrwp [N, K] -> Linear -> added to batch.x [N, out_dim]
    """
    def __init__(self, emb_dim, out_dim, use_bias=False, batchnorm=False,
                 layernorm=False, pe_name="rrwp"):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp = batch[f"{self.name}"]
        rrwp = self.fc(rrwp)

        if self.batchnorm:
            rrwp = self.bn(rrwp)

        if self.layernorm:
            rrwp = self.ln(rrwp)

        if "x" in batch:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp

        return batch


class RRWPLinearEdgeEncoder(torch.nn.Module):
    '''
        Merge RRWP with given edge-attr and Zero-padding to all pairs of node
        FC_1(RRWP) + FC_2(edge-attr)
        - FC_2 is the edge encoder applied before (LinearEdgeEncoder)
        - Zero-padding for non-existing edges in fully-connected graph

        Output: batch.edge_index becomes the union of
                (real edges + self-loops) U (RRWP pairs) [U (all pairs) if
                pad_to_full_graph], batch.edge_attr the summed features.
    '''
    def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False,
                 use_bias=False, pad_to_full_graph=True, fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on
        # providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr = overwrite_old_attr  # remove the old edge-attr

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn("batchnorm/layernorm might ruin some properties of "
                          "pe on providing shortest-path distance info ")

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = batch.rrwp_val
        edge_index = batch.edge_index
        edge_attr = batch.get('edge_attr', None)
        num_nodes = batch.num_nodes
        rrwp_val = self.fc(rrwp_val)

        if edge_attr is None:
            # zero features for the existing edges
            edge_attr = rrwp_val.new_zeros(edge_index.size(1), rrwp_val.size(1))

        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr,
                                                   num_nodes=num_nodes,
                                                   fill_value=0.)
            out_idx, out_val = coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val], dim=0),
                num_nodes, reduce='sum')

        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = coalesce(out_idx, out_val, num_nodes,
                                        reduce='sum')

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)

        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"
