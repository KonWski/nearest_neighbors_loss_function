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
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import Chem
import numpy as np

class SmileDataset(PygGraphPropPredDataset):

    def __init__(self, dataset_name, dataset_root, morgan_fingerprint: bool = False, 
                 rdkit_fp: bool = False, maccs_keys: bool = False, fpSize: int = 2048):
        PygGraphPropPredDataset.__init__(name = dataset_name, root = dataset_root)
        
        df_smiles = pd.read_csv(os.path.join(dataset_root, "ogbg_molhiv", "mapping", "mol.csv.gz"))
        self.data.smiles = list(df_smiles["smiles"])
        self.fpSize = fpSize

        self.data.morgan_fingerprints = None
        self.data.rdkit_fp = None
        self.data.maccs_keys = None
        self.faulty_indices = None

        if any(morgan_fingerprint, rdkit_fp, maccs_keys):

            mols, faulty_indices = self.__smiles_to_mols(self.data.smiles)
            self.faulty_indices = faulty_indices

            if morgan_fingerprint:
                self.data.morgan_fingerprints = self.__get_morgan_fingerprints(mols)

            if rdkit_fp:
                self.data.rdkit_fp = self.__get_rdkit_fps(mols)

            if maccs_keys:
                self.data.maccs_keys = self.__get_maccs_keys(mols)


    def __get_morgan_fingerprints(self, mols):
        
        fingerprints = []

        for mol in mols:
            if mol is not None:
                fingerprint = AllChem.GetFingerprint(mol, radius=3, fpSize=self.fpSize)
                fingerprints.append(fingerprint)
            else:
                fingerprints.append([0 for bit in range(self.fpSize)])    

        return np.array(fingerprints)


    def __get_rdkit_fps(self, mols):

        rdkbi = {}
        fingerprints = []

        for mol in mols:
            if mol is not None:
                fingerprint = Chem.RDKFingerprint(mol, maxPath = 5, fpSize=self.fpSize, bitInfo=rdkbi)
                fingerprints.append(fingerprint)
            else:
                fingerprints.append([0 for bit in range(self.fpSize)])    

        return np.array(fingerprints)


    def __get_maccs_keys(self, mols):

        fingerprints = []

        for mol in mols:
            if mol is not None:
                fingerprint = MACCSkeys.GenMACCSKeys(mol)
                fingerprints.append(fingerprint)
            else:
                fingerprints.append([0 for bit in range(167)])    

        return np.array(fingerprints)


    def __smiles_to_mols(self, smiles):

        faulty_indices = []
        mols = []

        for id, smile in enumerate(smiles):

            mol = Chem.MolFromSmiles(smile)

            if mol is None:
                faulty_indices.append(id)

            mols.append(mol)

        return mols, faulty_indices


    def get_idx_split(self):

        idx_split = super().get_idx_split()

        if self.faulty_indices is not None:

            faulty_index = torch.tensor(faulty_index)
            train_idx = idx_split["train"][~torch.isin(idx_split["train"], faulty_index)]
            valid_idx = idx_split["valid"][~torch.isin(idx_split["valid"], faulty_index)]
            test_idx = idx_split["test"][~torch.isin(idx_split["test"], faulty_index)]
            
            idx_split = {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx
            }

        return idx_split

    def __len__(self):
        return len()


def get_datasets(dataset_name: str, dataset_root: str, task_id: int, 
                 morgan_fingerprints: bool, rdkit_fp: bool, maccs_keys: bool,
                 train_augmentation_params: DataAugmentationParams, debug: bool):
    
    graph_augmentation = GraphAugmentation(train_augmentation_params)
    basic_graph_transformations = FloatTransformation()

    ogbg_dataset = SmileDataset(dataset_name, dataset_root, morgan_fingerprints, rdkit_fp, maccs_keys)
    ogbg_dataset.data.y = ogbg_dataset.data.y[:, task_id].unsqueeze(1)

    train_dataset = prepare_dataset(ogbg_dataset, "train", graph_augmentation, debug)
    valid_dataset = prepare_dataset(ogbg_dataset, "valid", basic_graph_transformations, debug)
    test_dataset = prepare_dataset(ogbg_dataset, "test", basic_graph_transformations, debug)

    return train_dataset, valid_dataset, test_dataset


def get_loaders(batch_size: int, dataset_name: str, task_id: int, dataset_root: str, 
                morgan_fingerprints: bool, rdkit_fp: bool, maccs_keys: bool, 
                train_augmentation_params: DataAugmentationParams, debug):

    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_name, dataset_root, task_id, morgan_fingerprints, rdkit_fp, 
                                                              maccs_keys, train_augmentation_params, debug)

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