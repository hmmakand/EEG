import torch.nn as nn
from torch_geometric.nn import (global_add_pool, global_max_pool,
                                global_mean_pool)

from ..act import act_dict

pooling_dict = {'add': global_add_pool,
                'mean': global_mean_pool,
                'max': global_max_pool}


class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    pool(x) -> [Linear(d/2^l -> d/2^(l+1)) + act] x L -> Linear(-> dim_out)

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
        graph_pooling (str): 'mean' | 'add' | 'max' (cfg.model.graph_pooling)
        act (str): activation name (cfg.gnn.act)
    """

    def __init__(self, dim_in, dim_out, L=2, graph_pooling='mean', act='relu'):
        super().__init__()
        self.pooling_fun = pooling_dict[graph_pooling]
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = act_dict[act]()

    def _apply_index(self, batch):
        return batch.graph_feature, getattr(batch, 'y', None)

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
