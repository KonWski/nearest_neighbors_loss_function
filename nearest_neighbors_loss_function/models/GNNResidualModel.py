from torch.nn import ReLU, Module, init, BatchNorm1d, Dropout
from torch_geometric.nn import Linear, MessagePassing, global_mean_pool


class GNNResidualModel(Module):

    def __init__(self, in_channels, hidden_dim, embedding_size):
        
        super().__init__()

        self.conv1 = GNNLayer(in_channels, hidden_dim)
        self.batch1 = BatchNorm1d(hidden_dim)
        self.conv2 = GNNLayer(hidden_dim, hidden_dim)
        self.batch2 = BatchNorm1d(hidden_dim)
        self.conv3 = GNNLayer(hidden_dim, hidden_dim)
        self.batch3 = BatchNorm1d(hidden_dim)

        self.conv3 = GNNLayer(hidden_dim, hidden_dim)
        self.batch3 = BatchNorm1d(hidden_dim)

        self.conv4 = GNNLayer(hidden_dim, hidden_dim)
        self.batch4 = BatchNorm1d(hidden_dim)

        self.batch5 = BatchNorm1d(hidden_dim)


        self.relu = ReLU()
        self.dropout = Dropout(p=0.2)
        self.linear = Linear(hidden_dim, embedding_size)
        self.embedding_size = embedding_size


    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch1(x)
        x = self.relu(x)
        x_layer1_out = self.dropout(x)

        x = self.conv2(x_layer1_out, edge_index, edge_attr)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x_layer2_out = x + x_layer1_out

        x = self.conv3(x_layer2_out, edge_index, edge_attr)
        x = self.batch3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x_layer3_out = x + x_layer2_out

        x = self.conv4(x_layer3_out, edge_index, edge_attr)
        x = self.batch4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x_layer4_out = x + x_layer3_out

        x = self.batch5(x_layer4_out)
        x = self.relu(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
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