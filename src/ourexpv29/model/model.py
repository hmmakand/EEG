import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from .config import ModelConfig
from .layers import GraphLayer, TemporalEncoder


class EEGGraphModel(nn.Module):
    """Two-class classifier for one graph of electrodes per sample.

    The logits are the sum of three paths:

        graph head        node and pair streams through GraphLayer x layers,
                          mean-pooled over electrodes and over pairs, then an MLP
        wide path         one linear layer on the flattened node and edge inputs,
                          so a specific electrode or electrode pair can be weighted
        temporal encoder  learned temporal and spatial filters on the signals

    Input (a batch of graphs, each with `num_electrodes` nodes in the same order):
        x          [N, node_features]   band powers; the log is taken here
        edge_index [2, E]               every ordered electrode pair, no self-loops
        batch      [N]                  graph of every node
        edge_attr  [E, edge_features]
        signal     [N, signal_samples]
    Output: logits [graphs, num_classes]
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        hidden, electrodes = config.hidden, config.num_electrodes

        self.node_encoder = nn.Linear(config.node_features, hidden)
        self.electrode_embedding = nn.Embedding(electrodes, hidden)
        self.edge_encoder = nn.Linear(config.edge_features, hidden)
        self.walk_node_encoder = nn.Linear(config.random_walk_steps, hidden, bias=False)
        self.walk_pair_encoder = nn.Linear(config.random_walk_steps, hidden, bias=False)
        self.layers = nn.ModuleList([
            GraphLayer(hidden, config.heads, config.dropout, config.attention_dropout, config.attention_clamp)
            for _ in range(config.layers)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(config.dropout),
                                  nn.Linear(hidden, config.num_classes))

        self.wide = nn.Linear(electrodes * config.node_features + electrodes ** 2 * config.edge_features,
                              config.num_classes)
        # The wide path starts at zero, so training begins from the other two paths.
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.wide.bias)
        self.wide_dropout = nn.Dropout(config.wide_dropout)

        self.temporal = TemporalEncoder(
            electrodes, config.signal_samples, config.num_classes, config.temporal_filters,
            config.temporal_kernel, config.spatial_filters, config.power_pool, config.power_stride,
            config.temporal_dropout)

    @torch.no_grad()
    def random_walks(self, weight, graph, source, target, num_graphs):
        """[I, M, ..., M^(K-1)] per graph, shape [graphs, n, n, K]; M is the
        row-normalised matrix of edge weights."""
        n = self.config.num_electrodes
        adjacency = weight.new_zeros(num_graphs, n, n)
        adjacency.index_put_((graph, source, target), weight, accumulate=True)
        step = adjacency / adjacency.sum(-1, keepdim=True).clamp(min=1e-12)
        power = torch.eye(n, device=weight.device).expand_as(adjacency)
        powers = [power]
        for _ in range(self.config.random_walk_steps - 1):
            power = power @ step
            powers.append(power)
        return torch.stack(powers, dim=-1)

    def forward(self, x, edge_index, batch, edge_attr, signal):
        config, n = self.config, self.config.num_electrodes
        num_nodes = x.size(0)
        num_graphs = num_nodes // n
        if num_nodes != num_graphs * n:
            raise ValueError(f"Every graph must have {n} electrodes")
        x = torch.log(x.clamp(min=1e-10))

        # Position of every node inside its graph, and all ordered pairs
        # (self pairs included) in the order graph, source, destination.
        local = torch.arange(num_nodes, device=x.device) - batch * n
        offsets = torch.arange(num_graphs, device=x.device).repeat_interleave(n * n) * n
        pair_source = torch.arange(n, device=x.device).repeat_interleave(n).repeat(num_graphs)
        pair_target = torch.arange(n, device=x.device).repeat(n * num_graphs)
        pair_index = torch.stack([offsets + pair_source, offsets + pair_target])
        pair_graph = offsets // n
        # Row of every edge in the pair stream.
        source, target = edge_index
        edge_position = batch[source] * n * n + local[source] * n + local[target]

        if set(config.paths) != {"graph", "wide", "temporal"}:
            return self.forward_selected_paths(x, edge_index, batch, edge_attr, signal, local, pair_index, pair_graph,
                                               edge_position, num_graphs)

        h = self.node_encoder(x) + self.electrode_embedding(local)
        temporal_logits = self.temporal(signal, num_graphs)

        pairs = h.new_zeros(num_graphs * n * n, config.hidden)
        pairs = pairs.index_add(0, edge_position, self.edge_encoder(edge_attr))
        dense_edges = edge_attr.new_zeros(num_graphs * n * n, config.edge_features)
        dense_edges = dense_edges.index_add(0, edge_position, edge_attr)

        walks = self.random_walks(edge_attr.abs().mean(-1), batch[source], local[source], local[target], num_graphs)
        h = h + self.walk_node_encoder(walks.diagonal(dim1=1, dim2=2).permute(0, 2, 1).reshape(num_nodes, -1))
        pairs = pairs + self.walk_pair_encoder(walks.reshape(num_graphs * n * n, -1))

        for layer in self.layers:
            h, pairs = layer(h, pairs, edge_index, edge_position, pair_index)

        pooled = torch.cat([global_mean_pool(h, batch, size=num_graphs),
                            global_mean_pool(pairs, pair_graph, size=num_graphs)], dim=-1)
        wide_input = torch.cat([x.reshape(num_graphs, -1), dense_edges.reshape(num_graphs, -1)], dim=-1)
        return self.head(pooled) + self.wide(self.wide_dropout(wide_input)) + temporal_logits

    def forward_selected_paths(self, x, edge_index, batch, edge_attr, signal, local, pair_index, pair_graph,
                               edge_position, num_graphs):
        """The same computation with only the paths of `config.paths` (an ablation). `x` is already the log power."""
        config, n = self.config, self.config.num_electrodes
        num_nodes = x.size(0)
        source, target = edge_index
        logits = 0
        if "graph" in config.paths:
            h = self.node_encoder(x) + self.electrode_embedding(local)
            pairs = h.new_zeros(num_graphs * n * n, config.hidden)
            pairs = pairs.index_add(0, edge_position, self.edge_encoder(edge_attr))
            walks = self.random_walks(edge_attr.abs().mean(-1), batch[source], local[source], local[target],
                                      num_graphs)
            h = h + self.walk_node_encoder(walks.diagonal(dim1=1, dim2=2).permute(0, 2, 1).reshape(num_nodes, -1))
            pairs = pairs + self.walk_pair_encoder(walks.reshape(num_graphs * n * n, -1))
            for layer in self.layers:
                h, pairs = layer(h, pairs, edge_index, edge_position, pair_index)
            pooled = torch.cat([global_mean_pool(h, batch, size=num_graphs),
                                global_mean_pool(pairs, pair_graph, size=num_graphs)], dim=-1)
            logits = logits + self.head(pooled)
        if "wide" in config.paths:
            dense_edges = edge_attr.new_zeros(num_graphs * n * n, config.edge_features)
            dense_edges = dense_edges.index_add(0, edge_position, edge_attr)
            wide_input = torch.cat([x.reshape(num_graphs, -1), dense_edges.reshape(num_graphs, -1)], dim=-1)
            logits = logits + self.wide(self.wide_dropout(wide_input))
        if "temporal" in config.paths:
            logits = logits + self.temporal(signal, num_graphs)
        return logits
