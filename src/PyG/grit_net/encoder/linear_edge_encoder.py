import torch


class LinearEdgeEncoder(torch.nn.Module):
    """Linear projection of continuous edge features: in_dim -> emb_dim.

    Original: `grit/encoder/linear_edge_encoder.py`, where in_dim was
    hard-coded per dataset (MNIST/CIFAR10: 1, attributed_triangle: 2). Here it is a constructor argument.
    """

    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.in_dim = in_dim
        self.encoder = torch.nn.Linear(in_dim, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch
