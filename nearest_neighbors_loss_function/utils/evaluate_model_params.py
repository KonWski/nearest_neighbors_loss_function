from dataclasses import dataclass

@dataclass
class EvaluateModelParams:
    evaluation_model_name: str = None
    
    knn_n_neighbors: int = None

    rf_n_estimators: int = None
    rf_min_samples_split: int = None
    rf_min_samples_leaf: int = None
    rf_criterion: str = None
    rf_max_depth: int = None
    rf_class_weight: str = None
