from dataclasses import dataclass
from typing import List

@dataclass
class TrainModelParams:
        seeds: List[int] = None 
        training_type: str = None 
        task_id: int = None   
        batch_size: int = None 
        triplet_loss_margin: float = None 
        batch_shaper_margin: float = None 
        gamma_recalculation_strategy: int = None  
        gamma_function: str = None 
        weight_distances: bool = None 
        focal_pow: float = None 
        density_awareness: bool = None 
        density_function: str = None 
        samples_difficultness: bool = None 
        lambda_samples_difficultness: float = None
        n_evaluation_models: int = None
        n_epochs: int = None
        save_path: str = None
        lr: float = None
        model_name: str = None
        model_in_channels: int = None
        model_hidden_channels: int = None
        model_n_blocks: int = None
        embedding_length: int = None
        optimized_param_name: str = None
        early_stop_window_size: int = None