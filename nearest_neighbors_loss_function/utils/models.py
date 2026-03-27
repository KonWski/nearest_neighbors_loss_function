import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class GNNModel(nn.Module):

    def __init__(self, in_channels, hidden_dim, embedding_size):
        
        super().__init__()

        self.conv1 = GNNLayer(in_channels, hidden_dim)
        self.conv2 = GNNLayer(hidden_dim, hidden_dim)
        self.conv3 = GNNLayer(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, embedding_size)
        self.embedding_size = embedding_size


    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr))
        x = self.relu(self.conv3(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        out = self.relu(self.linear(x))

        return out


class GNNLayer(MessagePassing):

    def __init__(self, in_channels, out_channels):

        super().__init__(aggr='mean')
        self.node_mlp = nn.Linear(in_channels, out_channels)
        self.edge_mlp = nn.Linear(3, out_channels)
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
        nn.init_xavier_uniform(self.node_mlp.weight)
        nn.init_xavier_uniform(self.edge_mlp.weight)