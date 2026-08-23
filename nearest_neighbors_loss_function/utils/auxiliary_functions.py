import random
import numpy as np
import torch
import os
import importlib
from torch_geometric import seed_everything
from dataclasses import asdict, is_dataclass
import json
from argparse import Namespace
from pathlib import Path

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

def create_hash_dir(path: str, experiment_hash: str):
    hash_dir_path = os.path.join(path, experiment_hash)
    os.makedirs(hash_dir_path)
    return hash_dir_path


def create_model_dir(experiment_dir_path: str, seed: int):
    model_dir_path = os.path.join(experiment_dir_path, str(seed))
    os.makedirs(model_dir_path)
    return model_dir_path


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


def save_conf(conf, path, hash, file_name_prefix):
    "Saves group of experiments / experiment configuration to a yaml file"

    if is_dataclass(conf):
        d_conf = asdict(conf)
    elif isinstance(conf, Namespace):
        d_conf = vars(conf)
    else:
        raise Exception("save_conf did not recognize the conf's class")

    d_conf["hash"] = hash
    conf_path = os.path.join(path, f"{file_name_prefix}_conf.json")

    with open(conf_path, "w") as f:
        json.dump(d_conf, f, indent=2)


def args_validation(args, workflow):

    if workflow == "train":
        pass
    
    elif workflow == "test":

        # augmentation
        if any([args.prob_add_gaussian_noise, 
                args.feature_noise_std,
                args.prob_mask_node_features,
                args.mask_node_share,
                args.prob_mask_edge_features,
                args.mask_edge_share
                ]):
            
            raise Exception("Data augmentation is turned on!")
        

def save_embeddings(seed_path, hash, train_embeddings, train_labels, valid_embeddings, 
                        valid_labels, test_embeddings, test_labels):
    
    torch.save(train_embeddings, os.path.join(seed_path, f"train_embeddings_{hash}.pt"))
    torch.save(train_labels, os.path.join(seed_path, f"train_labels_{hash}.pt"))
    
    torch.save(valid_embeddings, os.path.join(seed_path, f"valid_embeddings_{hash}.pt"))
    torch.save(valid_labels, os.path.join(seed_path, f"valid_labels_{hash}.pt"))

    torch.save(test_embeddings, os.path.join(seed_path, f"test_embeddings_{hash}.pt"))
    torch.save(test_labels, os.path.join(seed_path, f"test_labels_{hash}.pt"))


def find_model_path(seed, save_path):

    seed_path = Path(os.path.join(save_path, str(seed)))
    models = [path for path in seed_path.rglob("*.pt") if Path(path).stem[:6] == "model_"]
    n_models = len(models)

    if n_models > 1:
        raise Exception(f"Directory {seed_path} contains more than 1 model")

    model_path = models[0]
    model_hash = Path(models[0]).stem[6:]

    return model_path, model_hash, seed_path