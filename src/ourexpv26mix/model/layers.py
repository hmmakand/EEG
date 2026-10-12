"""Building blocks of the model.

    GatedGCN         message passing along the graph edges
    PairAttention    GRIT attention over all electrode pairs, conditioned on the pair stream
    GraphLayer       both of the above on one shared pair stream, plus a feed-forward block
    TemporalEncoder  learned temporal and spatial filters on the electrode signals
    NodeSignalEncoder  learned temporal filters on each electrode's own signal, as node features
    LowFrequencyEncoder  learned spatial filters on the low-frequency potentials

Every module takes and returns plain tensors.
"""
import torch
from torch import nn
from torch_geometric.utils import scatter, softmax


class GatedGCN(nn.Module):
    """Gated graph convolution over the edges (source j -> destination i).

        e_ij = D h_i + E h_j + C e_ij
        h_i  = A h_i + sum_j sigmoid(e_ij) * B h_j / sum_j sigmoid(e_ij)

    Returns the node and edge updates; the caller adds the residual.
    """

    def __init__(self, dim, dropout):
        super().__init__()
        self.A = nn.Linear(dim, dim)
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)
        self.D = nn.Linear(dim, dim)
        self.E = nn.Linear(dim, dim)
        self.bn_node = nn.BatchNorm1d(dim)
        self.bn_edge = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, e, edge_index):
        src, dst = edge_index
        e_ij = self.D(x)[dst] + self.E(x)[src] + self.C(e)
        gate = torch.sigmoid(e_ij)
        numerator = scatter(gate * self.B(x)[src], dst, 0, dim_size=x.size(0), reduce='sum')
        denominator = scatter(gate, dst, 0, dim_size=x.size(0), reduce='sum')
        x = self.A(x) + numerator / (denominator + 1e-6)
        x = self.dropout(self.act(self.bn_node(x)))
        e = self.dropout(self.act(self.bn_edge(e_ij)))
        return x, e


class PairAttention(nn.Module):
    """GRIT attention over node pairs (source j -> destination i), per head.

        s_ij = signed_sqrt((K_j + Q_i) * E_w) + E_b     conditioned on the pair vector
        e_ij = relu(s_ij)                               new pair vector
        a_ij = softmax_j(clamp(Aw . e_ij))
        h_i  = sum_j a_ij * V_j + VeRow . sum_j a_ij * e_ij
    """

    def __init__(self, dim, heads, clamp, dropout):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden={dim} must be divisible by heads={heads}")
        self.heads, self.head_dim, self.clamp = heads, dim // heads, clamp
        self.dropout = nn.Dropout(dropout)
        self.Q = nn.Linear(dim, dim, bias=True)
        self.K = nn.Linear(dim, dim, bias=False)
        self.E = nn.Linear(dim, dim * 2, bias=True)
        self.V = nn.Linear(dim, dim, bias=False)
        for layer in (self.Q, self.K, self.E, self.V):
            nn.init.xavier_normal_(layer.weight)
        self.Aw = nn.Parameter(torch.zeros(self.head_dim, heads, 1))
        nn.init.xavier_normal_(self.Aw)
        self.act = nn.ReLU()
        self.VeRow = nn.Parameter(torch.zeros(self.head_dim, heads, self.head_dim))
        nn.init.xavier_normal_(self.VeRow)

    def forward(self, x, e, pair_index):
        """x [nodes, dim], e [pairs, dim] -> node update [nodes, dim], pair update [pairs, dim]."""
        num_nodes = x.size(0)
        src, dst = pair_index
        Q = self.Q(x).view(-1, self.heads, self.head_dim)
        K = self.K(x).view(-1, self.heads, self.head_dim)
        V = self.V(x).view(-1, self.heads, self.head_dim)
        E = self.E(e).view(-1, self.heads, self.head_dim * 2)
        E_w, E_b = E[:, :, :self.head_dim], E[:, :, self.head_dim:]

        score = (K[src] + Q[dst]) * E_w
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        pair = self.act(score + E_b)

        score = torch.einsum("ehd,dhc->ehc", pair, self.Aw)
        score = torch.clamp(score, min=-self.clamp, max=self.clamp)
        attention = softmax(score, dst, num_nodes=num_nodes)
        weight = self.dropout(attention)

        h = scatter(V[src] * weight, dst, dim=0, dim_size=num_nodes, reduce='sum')
        row = scatter(pair * weight, dst, dim=0, dim_size=num_nodes, reduce='sum')
        h = h + torch.einsum("nhd,dhc->nhc", row, self.VeRow)
        return h.flatten(1), pair.flatten(1)


class GraphLayer(nn.Module):
    """GatedGCN and PairAttention on one shared pair stream.

        h_loc, e_loc = GatedGCN(h, e on the edges)
        h_att, e_att = PairAttention(h, e)
        h = Norm(h + h_loc) + Norm(h + O_h(h_att))
        e = Norm(e + O_e(e_att) + e_loc on the edges)
        h = Norm(h + FFN(h))                              FFN: d -> 2d -> d

    `e` holds one vector per ordered node pair of each graph (self pairs
    included); `edge_pos` is the row of every edge in that pair stream.
    """

    def __init__(self, dim, heads, dropout, attention_dropout, clamp):
        super().__init__()
        self.local = GatedGCN(dim, dropout)
        self.norm_local = nn.BatchNorm1d(dim)
        self.attention = PairAttention(dim, heads, clamp, attention_dropout)
        self.O_h = nn.Linear(dim, dim)
        self.O_e = nn.Linear(dim, dim)
        self.norm_attention = nn.BatchNorm1d(dim)
        self.norm_pair = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.ff1 = nn.Linear(dim, dim * 2)
        self.ff2 = nn.Linear(dim * 2, dim)
        self.act = nn.ReLU()
        self.norm_ff = nn.BatchNorm1d(dim)

    def forward(self, h, e, edge_index, edge_pos, pair_index):
        h_local, e_local = self.local(h, e[edge_pos], edge_index)
        h_local = self.norm_local(h + h_local)
        e_new = e.index_add(0, edge_pos, e_local)

        h_attention, e_attention = self.attention(h, e, pair_index)
        h_attention = self.norm_attention(h + self.dropout(self.O_h(h_attention)))
        e_new = e_new + self.dropout(self.O_e(e_attention))

        h = h_local + h_attention
        e = self.norm_pair(e_new)
        feed_forward = self.dropout(self.ff2(self.dropout(self.act(self.ff1(h)))))
        return self.norm_ff(h + feed_forward), e


class TemporalEncoder(nn.Module):
    """Learned temporal and spatial filters followed by log power over time.

        u = BN(Conv1d(signal))                    temporal filters, shared by all electrodes
        s = BN(spatial filters over electrodes)   one weight per (filter, temporal filter, electrode)
        f = log(avgpool(s^2))                     time-resolved log power
        logits = Linear(dropout(f))

    This is the ShallowConvNet idea (Schirrmeister et al., 2017): a spatial
    filter followed by log power is what CSP computes, with the filters learned.
    """

    def __init__(self, num_electrodes, signal_samples, num_classes, filters, kernel, spatial_filters,
                 pool, stride, dropout):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.conv = nn.Conv1d(1, filters, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(filters)
        self.pool = nn.AvgPool1d(pool, stride)
        self.dropout = nn.Dropout(dropout)
        steps = (signal_samples + 2 * (kernel // 2) - kernel + 1 - pool) // stride + 1
        if steps < 1:
            raise ValueError("signal_samples is too short for power_pool")
        self.spatial = nn.Parameter(torch.empty(spatial_filters, filters, num_electrodes))
        nn.init.xavier_uniform_(self.spatial)
        self.bn_spatial = nn.BatchNorm1d(spatial_filters)
        self.out = nn.Linear(spatial_filters * steps, num_classes)

    def forward(self, signal, num_graphs):
        """signal [graphs * electrodes, samples] -> logits [graphs, classes]."""
        u = self.bn(self.conv(signal.unsqueeze(1)))
        u = u.view(num_graphs, self.num_electrodes, *u.shape[1:])
        s = self.bn_spatial(torch.einsum('sfn,bnft->bst', self.spatial, u))
        log_power = torch.log(self.pool(s ** 2).clamp(min=1e-6))
        return self.out(self.dropout(log_power.flatten(1)))


class NodeSignalEncoder(nn.Module):
    """Node features from the electrode signals.

        u = BN(Conv1d(signal))            temporal filters, shared by all electrodes
        "mixed" only:
        s_i = BN(sum_j w_ij * u_j)        every node mixes the filtered signals of all nodes;
                                          w_ij = W[i, j] + Linear(edge features of the pair j -> i),
                                          `heads` mixtures per node
        p = log(mean over time of u^2)    "power": one value per filter
            log(avgpool(u^2))             "timed" and "mixed": one value per filter and time step
        with time attention:
        p = Transformer(p + position)     the time steps of a node attend to each other;
                                          a token is the node's filter values at one time step
        features = Linear(dropout(p))

    In "power" and "timed" an electrode's features come from that electrode
    alone. In "mixed" the electrodes are combined before the power is taken,
    as a spatial filter does, with weights given by the graph: a learned
    weight per pair of electrodes plus a term from the trial's edge features.
    """

    def __init__(self, mode, num_electrodes, signal_samples, dim, filters, kernel, pool, stride, dropout,
                 edge_features, heads, time_blocks=0, time_heads=4):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd")
        self.num_electrodes = num_electrodes
        self.conv = nn.Conv1d(1, filters, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(filters)
        self.mixed = mode == "mixed"
        if self.mixed:
            # Every mixture starts as the node's own signal plus small weights for the others.
            start = torch.eye(num_electrodes).repeat(heads, 1, 1) + 0.1 * torch.randn(heads, num_electrodes,
                                                                                     num_electrodes)
            self.pair_weight = nn.Parameter(start)
            self.edge_weight = nn.Linear(edge_features, heads, bias=False)
            self.bn_mixed = nn.BatchNorm1d(heads * filters)
            filters = heads * filters
        if mode == "power":
            steps = 1
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            steps = (signal_samples - pool) // stride + 1
            if steps < 1:
                raise ValueError("signal_samples is too short for power_pool")
            self.pool = nn.AvgPool1d(pool, stride)
        self.time_attention = None
        if time_blocks:
            if filters % time_heads:
                raise ValueError(f"The {filters} filters per time step must be divisible by node_time_heads")
            self.time_position = nn.Parameter(torch.zeros(steps, filters))
            block = nn.TransformerEncoderLayer(filters, time_heads, dim_feedforward=2 * filters, dropout=dropout,
                                               batch_first=True, norm_first=True)
            self.time_attention = nn.TransformerEncoder(block, time_blocks, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(filters * steps, dim)

    def forward(self, signal, pair_features=None):
        """signal [nodes, samples] -> [nodes, dim].

        `pair_features` [graphs, n, n, edge features], indexed source, destination; needed for "mixed".
        """
        u = self.bn(self.conv(signal.unsqueeze(1)))
        if self.mixed:
            n = self.num_electrodes
            u = u.view(-1, n, *u.shape[1:])                                           # [graphs, n, filters, samples]
            # weight[g, h, i, j]: share of source j in mixture h of destination i
            weight = self.pair_weight + self.edge_weight(pair_features).permute(0, 3, 2, 1)
            mixtures = torch.einsum('ghij,gjft->gihft', weight, u)
            u = self.bn_mixed(mixtures.reshape(-1, mixtures.shape[2] * mixtures.shape[3], mixtures.shape[4]))
        log_power = torch.log(self.pool(u ** 2).clamp(min=1e-6))
        if self.time_attention is not None:
            tokens = log_power.transpose(1, 2) + self.time_position          # [nodes, time steps, filters]
            log_power = self.time_attention(tokens).transpose(1, 2)
        return self.out(self.dropout(log_power.flatten(1)))


class LowFrequencyEncoder(nn.Module):
    """Classify from the slow potentials (for example 0.5-4 Hz).

        s = BN(spatial filters over electrodes)    one weight per (filter, electrode)
        logits = Linear(dropout(s))                on the time courses themselves

    Slow potentials are locked to the task in time and sign, so the time
    course is used as it is; no power is taken, unlike in TemporalEncoder.
    """

    def __init__(self, num_electrodes, samples, num_classes, filters, dropout):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.spatial = nn.Parameter(torch.empty(filters, num_electrodes))
        nn.init.xavier_uniform_(self.spatial)
        self.bn = nn.BatchNorm1d(filters)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(filters * samples, num_classes)
        nn.init.zeros_(self.out.weight)       # the path starts at zero, like the wide path
        nn.init.zeros_(self.out.bias)

    def forward(self, signal_low, num_graphs):
        """signal_low [graphs * electrodes, samples] -> logits [graphs, classes]."""
        signal_low = signal_low.view(num_graphs, self.num_electrodes, -1)
        components = self.bn(torch.einsum('fn,bnt->bft', self.spatial, signal_low))
        return self.out(self.dropout(components.flatten(1)))
