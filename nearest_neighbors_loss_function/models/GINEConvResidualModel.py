from torch.nn import ModuleList, Sequential, Linear, ReLU, Module, BatchNorm1d, Dropout
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GINEConv

class GINEConvResidualModel(Module):

    def __init__(self, in_channels, hidden_dim, embedding_size, n_blocks = 5):
        
        super().__init__()

        self.embedding_size = embedding_size
        self.layer_blocks = ModuleList()
        
        for i in range(n_blocks):
            
            layer_block = GINEBlock(in_channels, hidden_dim)
            self.layer_blocks.append(layer_block)

        self.batch = BatchNorm1d(hidden_dim)
        self.relu = ReLU()
        self.dropout = Dropout(p=0.2)
        self.linear = Linear(hidden_dim, embedding_size)


    def forward(self, data):
        
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        for layer_block in self.layer_blocks:
            print("Entered layer_block")
            x_layer_block = layer_block(x, edge_index, edge_attr)
            x = x + x_layer_block

        x = self.batch(x)
        x = self.relu(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        out = self.linear(x)

        return out


class GINEBlock(Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()

        self.mlp = Sequential(
            Linear(in_channels, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.gine_conv = GINEConv(self.mlp, edge_dim=3)
        self.batch_norm = BatchNorm1d(hidden_dim)
        self.relu = ReLU()
        self.dropout = Dropout(p=0.2)
        
    def forward(self, x, edge_index, edge_attr):

        x = self.gine_conv(x, edge_index, edge_attr)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x