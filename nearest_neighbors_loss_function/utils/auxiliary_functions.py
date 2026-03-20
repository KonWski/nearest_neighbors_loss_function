import random
import numpy as np
import torch
import os

def set_seed(seed: int):
    '''Set randomness for random, numpy, PyTorch CPU, PyTorch GPU, '''

    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_dir(path: str, experiment_hash: str):
    experiment_dir_path = os.path.join(path, experiment_hash)
    os.makedirs(experiment_dir_path)
    return experiment_dir_path


def create_model_dir(experiment_dir_path: str, seed: int):
    model_dir_path = os.path.join(experiment_dir_path, seed)
    os.makedirs(model_dir_path)
    return model_dir_path