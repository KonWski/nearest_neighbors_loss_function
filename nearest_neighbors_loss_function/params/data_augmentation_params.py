from dataclasses import dataclass

@dataclass
class DataAugmentationParams:
    prob_add_gaussian_noise: float = None
    feature_noise_std: float = None
    prob_mask_node_features: float = None
    mask_node_share: float = None
    prob_mask_edge_features: float = None
    mask_edge_share: float = None