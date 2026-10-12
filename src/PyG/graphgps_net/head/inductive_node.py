import torch.nn as nn

from ..act import act_dict


class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Equivalent of GraphGym's `MLP(new_layer_config(dim_in, dim_out,
    layers_post_mp, has_act=False, has_bias=True))`:
        [Linear(d, d) -> BatchNorm -> act] x (layers_post_mp - 1)
        -> Linear(d, dim_out)

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        layers_post_mp (int): number of linear layers (cfg.gnn.layers_post_mp)
        act (str): activation name (cfg.gnn.act)
    """

    def __init__(self, dim_in, dim_out, layers_post_mp=1, act='relu'):
        super().__init__()
        layers = []
        for _ in range(layers_post_mp - 1):
            layers += [nn.Linear(dim_in, dim_in, bias=False),
                       nn.BatchNorm1d(dim_in),
                       act_dict[act]()]
        layers.append(nn.Linear(dim_in, dim_out, bias=True))
        self.layer_post_mp = nn.Sequential(*layers)

    def _apply_index(self, batch):
        return batch.x, getattr(batch, 'y', None)

    def forward(self, batch):
        batch.x = self.layer_post_mp(batch.x)
        pred, label = self._apply_index(batch)
        return pred, label
