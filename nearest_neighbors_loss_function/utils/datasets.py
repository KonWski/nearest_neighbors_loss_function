from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from nearest_neighbors_loss_function.utils.balanced_sampler import BalancedSampler
from nearest_neighbors_loss_function.utils.transformations import FloatTransformation, GraphAugmentation
from nearest_neighbors_loss_function.params.data_augmentation_params import DataAugmentationParams
import math
from torch_geometric.loader import DataLoader
import torch
import logging
import os
import pandas as pd

class SmileDataset(PygGraphPropPredDataset):

    def __init__(self, dataset_name, dataset_root):
        PygGraphPropPredDataset.__init__(name = dataset_name, root = dataset_root)
        
        df_smiles = pd.read_csv(os.path.join(dataset_root, "ogbg_molhiv", "mapping", "mol.csv.gz"))
        self.data.smiles = list(df_smiles["smiles"])


def get_datasets(dataset_name: str, dataset_root: str, task_id: int, 
                 train_augmentation_params: DataAugmentationParams, debug: bool):
    
    graph_augmentation = GraphAugmentation(train_augmentation_params)
    basic_graph_transformations = FloatTransformation()

    ogbg_dataset = SmileDataset(name = dataset_name, root = dataset_root)
    ogbg_dataset.data.y = ogbg_dataset.data.y[:, task_id].unsqueeze(1)

    train_dataset = prepare_dataset(ogbg_dataset, "train", graph_augmentation, debug)
    valid_dataset = prepare_dataset(ogbg_dataset, "valid", basic_graph_transformations, debug)
    test_dataset = prepare_dataset(ogbg_dataset, "test", basic_graph_transformations, debug)

    return train_dataset, valid_dataset, test_dataset


def get_loaders(batch_size: int, dataset_name: str, task_id: int, dataset_root: str, 
                train_augmentation_params: DataAugmentationParams, debug):

    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_name, dataset_root, task_id, train_augmentation_params, debug)

    n_batches = math.ceil(len(train_dataset) / batch_size)
    n_train_minority_samples = int(train_dataset.y.sum())
    minority_per_batch = int(n_train_minority_samples / n_batches)
    balanced_sampler = BalancedSampler(train_dataset.y, n_train_minority_samples, 1, minority_per_batch, batch_size, n_batches)

    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def prepare_dataset(ogbg_dataset: SmileDataset, phase: str, transformation, debug):

    # find not nan labels
    phase_indices = ogbg_dataset.get_idx_split()[phase]
    n_obs_before_filter = len(phase_indices)
    labels = ogbg_dataset[phase_indices].y
    not_nan_indices = ~torch.isnan(labels).any(dim=1)
    phase_indices = phase_indices[not_nan_indices]

    if debug:
        dataset = ogbg_dataset[phase_indices[:10000]]
    else:
        dataset = ogbg_dataset[phase_indices]

    dataset.transform = transformation

    logging.info(f"{phase}_dataset, n_obs_before_filter: {n_obs_before_filter},  n_obs_after_filter: {len(dataset)}, n_minority_class: {dataset.y.sum()}")

    return dataset


def turn_off_augmentations(dataloader):
    dataloader.dataset.transform = FloatTransformation()
    return dataloader