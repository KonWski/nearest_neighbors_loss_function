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

    def __init__(self, dataset_name, dataset_root, use_morgan_fingerprint: bool = False, 
                 use_rdkit_fp: bool = False, use_maccs_keys: bool = False, fpSize: int = 2048):
        PygGraphPropPredDataset.__init__(self, name = dataset_name, root = dataset_root)
        
        df_smiles = pd.read_csv(os.path.join(dataset_root, "ogbg_molhiv", "mapping", "mol.csv.gz"))
        self.smiles = list(df_smiles["smiles"])
        self.fpSize = fpSize

        self.use_morgan_fingerprints = use_morgan_fingerprint
        self.use_rdkit_fp = use_rdkit_fp
        self.use_maccs_keys = use_maccs_keys

        self.morgan_fingerprints = None
        self.rdkit_fp = None
        self.maccs_keys = None

        self.faulty_indices = None
        self.embedding_extra_length = 0
        self.use_extra_embeddings = any([use_morgan_fingerprint, use_rdkit_fp, use_maccs_keys])
        self.extra_embeddings_methods = []

        if self.use_extra_embeddings:

            mols, faulty_indices = self.__smiles_to_mols(self.smiles)
            self.faulty_indices = faulty_indices

            if use_morgan_fingerprint:
                self.morgan_fingerprints = self.__get_morgan_fingerprints(mols)
                self.embedding_extra_length += self.morgan_fingerprints.shape[1]
                self.extra_embeddings_methods.append("morgan_fingerprints")

            if use_rdkit_fp:
                self.rdkit_fp = self.__get_rdkit_fps(mols)
                self.embedding_extra_length += self.rdkit_fp.shape[1]
                self.extra_embeddings_methods.append("rdkit_fp")

            if use_maccs_keys:
                self.maccs_keys = self.__get_maccs_keys(mols)
                self.embedding_extra_length += self.maccs_keys.shape[1]
                self.extra_embeddings_methods.append("maccs_keys")
            
            self.__rearrange_data()

    def __get_morgan_fingerprints(self, mols):
        
        fingerprints = []

        for mol in mols:
            if mol is not None:
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=self.fpSize)
                fingerprints.append(fingerprint)
            else:
                fingerprints.append([0 for _ in range(self.fpSize)])    

        return torch.tensor(fingerprints)


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

        return mols, torch.tensor(faulty_indices)


    def __rearrange_data(self):

        data_list = []

        for i in range(len(self)):
            data = self[i]

            for method in self.extra_embeddings_methods:
                data[method] = getattr(self, method)

            data_list.append(data)

        self.data, self.slices = self.collate(data_list)


    def get_idx_split(self):

        idx_split = super().get_idx_split()

        if self.faulty_indices is not None:

            train_idx = idx_split["train"][~torch.isin(idx_split["train"], self.faulty_indices)]
            valid_idx = idx_split["valid"][~torch.isin(idx_split["valid"], self.faulty_indices)]
            test_idx = idx_split["test"][~torch.isin(idx_split["test"], self.faulty_indices)]
            
            idx_split = {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx
            }

        return idx_split


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