from torch.nn import Sequential, Linear, ReLU
from .GINEConvResidualModel import GINEConvResidualModel
import torch

class GINEConvResidualFocalModel(GINEConvResidualModel):

    def __init__(self, in_channels, hidden_dim, n_blocks, embedding_size):

        super().__init__(in_channels, hidden_dim, n_blocks, embedding_size)

        self.classifier = Sequential(
            Linear(embedding_size, 1024),
            ReLU(),
            Linear(1024, 1)
        )
    
    def classify(self, x, y):
        diff = torch.abs(x - y)
        logit = self.classifier(diff).squeeze(1)
        return logit