from torch.nn import ModuleList, Sequential, Linear, ReLU, Module, BatchNorm1d, Dropout, init
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GINEConv
import torch.nn.functional as F

class GINEConvResidualModel(Module):

    def __init__(self, in_channels, hidden_dim, embedding_size, n_blocks = 4):

        super().__init__()

        self.embedding_size = embedding_size
        self.node_encoder = Sequential(
            Linear(in_channels, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.node_encoder.apply(self._init_weights)

        self.edge_encoder = Linear(3, hidden_dim)
        self._init_weights(self.edge_encoder)

        self.layer_blocks = ModuleList()
        for _ in range(n_blocks):            
            layer_block = GINEBlock(hidden_dim)
            layer_block.apply(self._init_weights)
            self.layer_blocks.append(layer_block)

        self.batch = BatchNorm1d(hidden_dim)
        self.relu = ReLU()
        self.dropout = Dropout(p=0.2)
        self.linear = Linear(hidden_dim, embedding_size)
        self._init_weights(self.linear)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, data):

        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer_block in self.layer_blocks:
            x_layer_block = layer_block(x, edge_index, edge_attr)
            x = x + x_layer_block

        x = self.batch(x)
        x = self.relu(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        out = self.linear(x)

        return out


class GINEBlock(Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.gine_conv = GINEConv(self.mlp, edge_dim=hidden_dim)
        self.batch_norm = BatchNorm1d(hidden_dim)
        self.relu = ReLU()
        self.dropout = Dropout(p=0.2)

    def forward(self, x, edge_index, edge_attr):

        x = self.gine_conv(x, edge_index, edge_attr)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x