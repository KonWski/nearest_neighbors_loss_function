from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from nearest_neighbors_loss_function.utils.balanced_sampler import BalancedSampler
import math
from torch_geometric.loader import DataLoader
import torch
import logging

def get_datasets(dataset_name = "ogbg-molhiv", dataset_root = 'dataset/', task_id=0, debug=False):
    ogbg_dataset = PygGraphPropPredDataset(name = dataset_name, root = dataset_root)

    train_dataset = prepare_dataset(ogbg_dataset, task_id, "train", debug)
    valid_dataset = prepare_dataset(ogbg_dataset, task_id, "valid", debug)
    test_dataset = prepare_dataset(ogbg_dataset, task_id, "test", debug)

    return train_dataset, valid_dataset, test_dataset

def get_loaders(batch_size, dataset_name, task_id, dataset_root = 'dataset/', debug=False):

    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_name, dataset_root, task_id, debug)

    n_batches = math.ceil(len(train_dataset) / batch_size)
    n_train_minority_samples = int(train_dataset.y.sum())
    minority_per_batch = int(n_train_minority_samples / n_batches)

    print(f"n_train_minority_samples: {n_train_minority_samples}")
    print(f"minority_per_batch: {minority_per_batch}")
    print(f"batch_size: {batch_size}")
    print(f"n_batches: {n_batches}")
    balanced_sampler = BalancedSampler(train_dataset.y, n_train_minority_samples, 1, minority_per_batch, batch_size, n_batches)

    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def prepare_dataset(ogbg_dataset, task_id, phase, debug):

    # find not nan labels
    phase_indices = ogbg_dataset.get_idx_split()[phase]
    n_obs_before_filter = len(phase_indices)
    labels = ogbg_dataset[phase_indices].y[:,task_id].unsqueeze(1)
    not_nan_indices = ~torch.isnan(labels).any(dim=1)
    phase_indices = phase_indices[not_nan_indices]

    if debug:
        dataset = ogbg_dataset[phase_indices[:10000]]
    else:
        dataset = ogbg_dataset[phase_indices]
    
    dataset.y = dataset.y[:,task_id].unsqueeze(1)
    print(f"dataset.y: {dataset.y}")
    print(f"dataset.y.shape: {dataset.y.shape}")
    
    logging.info(f"{phase}_dataset, n_obs_before_filter: {n_obs_before_filter},  n_obs_after_filter: {len(dataset)}, n_minority_class: {dataset.y.sum()}")

    return dataset