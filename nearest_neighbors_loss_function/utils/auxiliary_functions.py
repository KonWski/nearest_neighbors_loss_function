import random
import numpy as np
import torch
import importlib
from nearest_neighbors_loss_function.models import GINEConvEncoderResidualModel
from torch_geometric import seed_everything

def set_seed(seed: int):
    '''Set randomness for random, numpy, PyTorch CPU, PyTorch GPU, '''

    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def adjust_graph_data_dtype(data, model):

    if not isinstance(model, GINEConvEncoderResidualModel):
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()

    return data


def ignore_top_weight(distances):
    weights = 1 / (distances + 1e-9)
    max_idx = np.argmax(weights, axis=1)
    weights[np.arange(weights.shape[0]), max_idx] = 0

    return weights


def get_model(model_name, in_channels, hidden_dim, model_n_blocks, embedding_size):

    module_name = "nearest_neighbors_loss_function.models"
    module = importlib.import_module(module_name)
    cls = getattr(module, model_name)
    model = cls(in_channels, hidden_dim, model_n_blocks, embedding_size)

    return model