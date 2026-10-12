import torch


class LinearNodeEncoder(torch.nn.Module):
    """Linear projection of continuous node features: dim_in -> emb_dim.

    Original: `graphgps/encoder/linear_node_encoder.py` (dim_in came from
    cfg.share.dim_in; here it is passed explicitly).
    """

    def __init__(self, dim_in, emb_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(dim_in, emb_dim)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch
