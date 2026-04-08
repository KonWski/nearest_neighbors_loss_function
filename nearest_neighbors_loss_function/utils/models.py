import torch.nn.functional as F
from torch.nn import ReLU, Module, init, BatchNorm1d
from torch_geometric.nn import GCNConv, Linear, MessagePassing, global_mean_pool, SAGEConv, GATConv
import torch
import copy

class GNNModel(Module):

    def __init__(self, in_channels, hidden_dim, embedding_size):
        
        super().__init__()

        self.conv1 = GNNLayer(in_channels, hidden_dim)
        self.batch1 = BatchNorm1d(hidden_dim)
        self.conv2 = GNNLayer(hidden_dim, hidden_dim)
        self.batch2 = BatchNorm1d(hidden_dim)
        self.conv3 = GNNLayer(hidden_dim, hidden_dim)
        self.batch3 = BatchNorm1d(hidden_dim)
        self.relu = ReLU()
        self.linear = Linear(hidden_dim, embedding_size)
        self.embedding_size = embedding_size


    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batch3(x)
        x = self.relu(x)

        x = global_mean_pool(x, batch)
        out = self.linear(x)

        return out

class GNNResidualModel(Module):

    def __init__(self, in_channels, hidden_dim, embedding_size):
        
        super().__init__()

        self.conv1 = GNNLayer(in_channels, hidden_dim)
        self.batch1 = BatchNorm1d(hidden_dim)
        self.conv2 = GNNLayer(hidden_dim, hidden_dim)
        self.batch2 = BatchNorm1d(hidden_dim)
        self.conv3 = GNNLayer(hidden_dim, hidden_dim)
        self.batch3 = BatchNorm1d(hidden_dim)
        self.relu = ReLU()
        self.linear = Linear(hidden_dim, embedding_size)
        self.embedding_size = embedding_size


    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch1(x)
        x = self.relu(x)
        x_layer1_out = copy.deepcopy(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.conv3(x, edge_index, edge_attr) + x_layer1_out
        x = self.batch3(x)
        x = self.relu(x)

        x = global_mean_pool(x, batch)
        out = self.linear(x)

        return out


class GNNLayer(MessagePassing):

    def __init__(self, in_channels, out_channels):

        super().__init__(aggr='mean')
        self.node_mlp = Linear(in_channels, out_channels)
        self.edge_mlp = Linear(3, out_channels)
        self._initialize_weights()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        x_j = self.node_mlp(x_j)
        
        if edge_attr is not None:
            edge_emb = self.edge_mlp(edge_attr)
            return x_j + edge_emb

        return x_j

    def update(self, aggr_out):
        return aggr_out
    
    def _initialize_weights(self):
        init.xavier_uniform_(self.node_mlp.weight)
        init.xavier_uniform_(self.edge_mlp.weight)


class GCN(Module):
    def __init__(self, in_channels, hidden_dim, embedding_size):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = Linear(hidden_dim, embedding_size)

    def forward(self, data):
        x, edge_index, _, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)    
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x


class GraphSAGE(Module):
    def __init__(self, in_channels, hidden_dim, embedding_size):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.linear = Linear(hidden_dim, embedding_size)

    def forward(self, data):
        x, edge_index, _, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.linear(x)

        return x


class GAT(Module):
    def __init__(self, in_channels, hidden_dim, embedding_size, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.linear = Linear(hidden_dim, embedding_size)

    def forward(self, data):
        x, edge_index, _, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.linear(x)

        return x