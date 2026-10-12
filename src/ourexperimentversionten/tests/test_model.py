"""The version-ten GATv2 classifier is used unchanged."""
import unittest

import torch
from torch_geometric.data import Batch, Data

from ..model import GAT


class ModelTests(unittest.TestCase):
    def test_output_shape_gradients_and_empty_graphs(self):
        torch.manual_seed(0)
        model = GAT(hidden_channels=22, heads=3, in_channels=6)
        full = torch.tensor([[i, j] for i in range(22) for j in range(22) if i != j]).T
        graphs = [Data(x=torch.randn(22, 6), edge_index=full),
                  Data(x=torch.randn(22, 6), edge_index=torch.empty((2, 0), dtype=torch.long))]  # no pair above threshold
        batch = Batch.from_data_list(graphs)
        out = model(batch.x, batch.edge_index, batch.batch)
        self.assertEqual(tuple(out.shape), (2, 2))
        out.sum().backward()
        self.assertEqual([name for name, p in model.named_parameters() if p.grad is None], [])


if __name__ == "__main__":
    unittest.main()
