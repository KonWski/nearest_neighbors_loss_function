from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from nearest_neighbors_loss_function.utils.balanced_sampler import BalancedSampler
import math
from torch_geometric.loader import DataLoader
import torch
import logging
from torch.utils.data import Subset

def get_datasets(dataset_name = "ogbg-molhiv", dataset_root = 'dataset/', debug=False):
    ogbg_dataset = PygGraphPropPredDataset(name = dataset_name, root = dataset_root)

    if debug:
        train_dataset = ogbg_dataset[ogbg_dataset.get_idx_split()["train"][:10000]]
        valid_dataset = ogbg_dataset[ogbg_dataset.get_idx_split()["valid"][:10000]]
        test_dataset = ogbg_dataset[ogbg_dataset.get_idx_split()["test"][:10000]]
    else:
        train_dataset = ogbg_dataset[ogbg_dataset.get_idx_split()["train"]]
        valid_dataset = ogbg_dataset[ogbg_dataset.get_idx_split()["valid"]]
        test_dataset = ogbg_dataset[ogbg_dataset.get_idx_split()["test"]]

    return train_dataset, valid_dataset, test_dataset

def get_loaders(batch_size, dataset_name, task_id, dataset_root = 'dataset/', debug=False):

    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_name, dataset_root, debug)

    train_dataset = prepare_dataset(train_dataset, task_id, "train")
    valid_dataset = prepare_dataset(valid_dataset, task_id, "valid")
    test_dataset = prepare_dataset(test_dataset, task_id, "test")

    n_batches = math.ceil(len(train_dataset) / batch_size)
    n_train_minority_samples = train_dataset.y.sum()
    minority_per_batch = int(n_train_minority_samples / n_batches)
    balanced_sampler = BalancedSampler(train_dataset.y, n_train_minority_samples, 1, minority_per_batch, batch_size, n_batches)

    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def prepare_dataset(dataset, task_id, phase):
    
    # select task
    dataset.y = dataset.y[:,task_id].unsqueeze(1)    
    
    # filter out nans
    not_nan_indices = ~torch.isnan(dataset.y).any(dim=1)
    # dataset.y = dataset.y[not_nan_indices]
    # dataset.x = dataset.x[not_nan_indices]
    
    subset = Subset(dataset, not_nan_indices)
    logging.info(f"{phase}_dataset len: {len(dataset)}, n_minority_class: {dataset.y.sum()}")

    return subset