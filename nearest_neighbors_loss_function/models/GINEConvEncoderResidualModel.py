from torch.nn import ModuleList, Sequential, Linear, ReLU, Module, BatchNorm1d, Dropout, init
from torch_geometric.nn import global_mean_pool, GraphNorm
from torch_geometric.nn.conv import GINEConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn.models import MLP

class GINEConvEncoderResidualModel(Module):

    def __init__(self, in_channels, hidden_dim, n_blocks, embedding_size):

        super().__init__()

        self.embedding_size = embedding_size
        self.node_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.edge_encoder = BondEncoder(emb_dim=hidden_dim)
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
            x = layer_block(x, edge_index, edge_attr, batch)

        x = self.batch(x)
        x = self.relu(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        out = self.linear(x)

        return out


class GINEBlock(Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.batch_norm = GraphNorm(hidden_dim)
        self.relu = ReLU()
        gine_conv_mlp = MLP(num_layers=2, in_channels=hidden_dim, hidden_channels=hidden_dim, 
                       out_channels=hidden_dim, batch_norm=True)
        self.gine_conv = GINEConv(gine_conv_mlp, edge_dim=hidden_dim)
        self.dropout = Dropout(p=0.2)

    def forward(self, x, edge_index, edge_attr, batch):

        res = x
        x = self.batch_norm(x, batch)
        x = self.relu(x)
        x = self.gine_conv(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = x + res

        return x