from torch_geometric.transforms import BaseTransform
import torch
import numpy as np
from nearest_neighbors_loss_function.params.data_augmentation_params import DataAugmentationParams


class GraphAugmentation(BaseTransform):
  
    def __init__(self, params: DataAugmentationParams):

        self.prob_add_gaussian_noise = params.prob_add_gaussian_noise
        self.feature_noise_std = params.feature_noise_std
        self.prob_mask_node_features = params.prob_mask_node_features
        self.mask_node_share = params.mask_node_share
        self.prob_mask_edge_features = params.prob_mask_edge_features
        self.mask_edge_share = params.mask_edge_share


    def forward(self, data):

        data = data.clone()
        data.x = data.x.to(torch.float32)

        if self.prob_add_gaussian_noise > 0 and np.random.uniform(0, 1) <= self.prob_add_gaussian_noise:
            data = self.add_gaussian_noise(data)

        if self.prob_mask_node_features > 0 and np.random.uniform(0, 1) <= self.prob_mask_node_features:
            data = self.mask_node_features(data)

        if self.prob_mask_edge_features > 0 and np.random.uniform(0, 1) <= self.prob_mask_edge_features:
            data = self.mask_edge_features(data)

        return data


    def add_gaussian_noise(self, data):
        noise = torch.normal(torch.zeros(data.x.shape), std=self.feature_noise_std).to(data.x.device)
        data.x = torch.clamp(data.x + noise, min=0.0)

        return data
    

    def mask_node_features(self, data):
        n_nodes = data.x.shape[0]
        n_nodes_masked = int(n_nodes * self.mask_node_share)

        node_indices_masked = torch.randperm(n_nodes)[:n_nodes_masked]
        data.x[node_indices_masked] = 0.0

        return data
    

    def mask_edge_features(self, data):
        n_edges = data.edge_attr.shape[0]
        n_edges_masked = int(n_edges * self.mask_edge_share)

        edge_indices_masked = torch.randperm(n_edges)[:n_edges_masked]
        data.edge_attr[edge_indices_masked] = 0.0

        return data


class FloatTransformation(BaseTransform):
  
    def forward(self, data):
        data.x = data.x.to(torch.float32)
        return data