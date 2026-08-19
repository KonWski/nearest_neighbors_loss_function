import torch
from .datasets import get_loaders
from .train import train_triplet
from .evaluate_model import test_best_seed_model, test_model
import logging
from nearest_neighbors_loss_function.utils.statistics import Statistics
from nearest_neighbors_loss_function.params.evaluate_model_params import EvaluateModelParams
from nearest_neighbors_loss_function.params.train_param_holder import TrainParamHolder
from nearest_neighbors_loss_function.utils.auxiliary_functions import create_hash_dir, save_conf, \
    args_validation, save_embeddings
from nearest_neighbors_loss_function.params.data_augmentation_params import DataAugmentationParams
import os
from pathlib import Path
from uuid import uuid4
from nearest_neighbors_loss_function.utils.generate_embeddings import generate_all_splits_embeddings
from pathlib import Path

def train_workflow(args):
    
    log_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_augmentation_params = DataAugmentationParams(
        prob_add_gaussian_noise = args.prob_add_gaussian_noise,
        feature_noise_std = args.feature_noise_std,
        prob_mask_node_features = args.prob_mask_node_features,
        mask_node_share = args.mask_node_share,
        prob_mask_edge_features = args.prob_mask_edge_features,
        mask_edge_share = args.mask_edge_share,
    )

    train_loader, valid_loader, test_loader = get_loaders(args.batch_size, args.dataset_name, args.task_id, 
                                                          args.dataset_root, False, False, False, 
                                                          train_augmentation_params, debug=args.debug)
    
    evaluate_model_params = EvaluateModelParams(
        evaluation_model_name="knn",
        knn_n_neighbors=args.knn_n_neighbors
    )

    train_param_holder = TrainParamHolder(args)

    group_experiment_hash = uuid4().hex
    group_experiment_save_path = create_hash_dir(args.save_path, group_experiment_hash)
    save_conf(args, group_experiment_save_path, group_experiment_hash, "group_experiment")

    for train_params in train_param_holder:

        experiment_hash = uuid4().hex
        experiment_dir_path = create_hash_dir(group_experiment_save_path, experiment_hash)
        save_conf(train_params, experiment_dir_path, experiment_hash, "experiment")

        statistics, best_seed_models, n_train_samples = train_triplet(
            seeds=train_params.seeds,
            train_loader=train_loader,
            valid_loader=valid_loader,
            training_type=train_params.training_type,
            batch_size=train_params.batch_size,
            task_id=train_params.task_id,
            triplet_loss_margin=train_params.triplet_loss_margin,
            batch_shaper_margin=train_params.batch_shaper_margin,
            gamma_recalculation_strategy=train_params.gamma_recalculation_strategy,
            gamma_function=train_params.gamma_function,
            weight_distances=train_params.weight_distances,
            focal_pow=train_params.focal_pow,
            density_awareness=train_params.density_awareness,
            density_function=train_params.density_function,
            samples_difficultness=train_params.samples_difficultness,
            lambda_samples_difficultness=train_params.lambda_samples_difficultness,
            n_evaluation_models=train_params.n_evaluation_models,
            evaluate_model_params=evaluate_model_params,
            n_epochs=train_params.n_epochs,
            experiment_dir_path=experiment_dir_path,
            experiment_hash=experiment_hash,
            lr=train_params.lr,
            model_name=train_params.model_name,
            model_in_channels=train_params.model_in_channels,
            model_hidden_channels=train_params.model_hidden_channels,
            model_n_blocks=train_params.model_n_blocks,
            embedding_length=train_params.embedding_length,
            optimized_param_name=train_params.optimized_param_name,
            early_stop_window_size=train_params.early_stop_window_size,
            device=device
        )

        statistics = test_best_seed_model(
            train_params.model_name,
            statistics, 
            best_seed_models, 
            train_params.model_in_channels, 
            train_params.model_hidden_channels,
            train_params.model_n_blocks,
            train_params.embedding_length, 
            train_loader,
            n_train_samples,
            test_loader,
            evaluate_model_params,
            "test", 
            device
        )

        statistics.save()


def test_workflow(args):

    log_args(args)
    args_validation(args, args.workflow)
    experiment_hash = args.save_path.split("/")[-1]
    statistics = Statistics(args.n_epochs, args.save_path, experiment_hash, args.optimized_param_name, -1, "test_report")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_augmentation_params = DataAugmentationParams(
        prob_add_gaussian_noise = args.prob_add_gaussian_noise,
        feature_noise_std = args.feature_noise_std,
        prob_mask_node_features = args.prob_mask_node_features,
        mask_node_share = args.mask_node_share,
        prob_mask_edge_features = args.prob_mask_edge_features,
        mask_edge_share = args.mask_edge_share,
    )

    train_loader, valid_loader, test_loader = get_loaders(args.batch_size, args.dataset_name, args.task_id, 
                                                          args.dataset_root, args.morgan_fingerprints, args.rdkit_fp, 
                                                          args.maccs_keys, train_augmentation_params, debug=False)
    n_train_samples = len(train_loader.dataset)
    n_valid_samples = len(valid_loader.dataset)
    n_test_samples = len(test_loader.dataset)

    evaluate_model_params = EvaluateModelParams(
        evaluation_model_name=args.evaluation_model_name,
        knn_n_neighbors=args.knn_n_neighbors,
        rf_n_estimators=args.rf_n_estimators,
        rf_min_samples_split=args.rf_min_samples_split,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        rf_criterion=args.rf_criterion,
        rf_max_depth=args.rf_max_depth,
        rf_class_weight=args.rf_class_weight
    )

    for seed in os.listdir(args.save_path):

        # double check if its the seed dir
        try:
            int(seed)
        except ValueError:
            continue
        
        seed_path = Path(os.path.join(args.save_path, str(seed)))
        models = list(seed_path.rglob("*.pt"))
        n_models = len(models)

        if n_models > 1:
            raise Exception(f"Directory {seed_path} contains more than 1 model")

        statistics = test_model(str(models[0]), args.model_name, args.model_in_channels, args.model_hidden_channels, args.model_n_blocks, args.embedding_length, 
                                train_loader, n_train_samples, valid_loader, n_valid_samples, test_loader, n_test_samples, 
                                evaluate_model_params, statistics, int(seed), device)

    statistics.save()


def generate_embeddings_workflow(args):

    log_args(args)
    # args_validation(args, args.workflow)
    experiment_hash = args.save_path.split("/")[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = get_loaders(args.batch_size, args.dataset_name, args.task_id, 
                                                          args.dataset_root, args.morgan_fingerprints, args.rdkit_fp, 
                                                          args.maccs_keys, None, debug=False)

    n_train_samples = len(train_loader.dataset)
    n_valid_samples = len(valid_loader.dataset)
    n_test_samples = len(test_loader.dataset)

    for seed in os.listdir(args.save_path):

        # double check if its the seed dir
        try:
            int(seed)
        except ValueError:
            continue
        
        seed_path = Path(os.path.join(args.save_path, str(seed)))
        models = list(seed_path.rglob("*.pt"))
        model_hash = Path(models[0]).stem
        n_models = len(models)

        if n_models > 1:
            raise Exception(f"Directory {seed_path} contains more than 1 model")

        train_embeddings, train_labels, valid_embeddings, valid_labels, test_embeddings, test_labels = \
            generate_all_splits_embeddings(
                seed, 
                train_loader, 
                n_train_samples, 
                valid_loader, 
                n_valid_samples, 
                test_loader,
                n_test_samples, 
                str(models[0]), 
                args.model_name, 
                args.model_in_channels, 
                args.model_hidden_channels,
                args.model_n_blocks, 
                args.embedding_length, 
                device
            )

        save_embeddings(seed_path, model_hash, train_embeddings, train_labels, valid_embeddings, 
                        valid_labels, test_embeddings, test_labels)

        if n_models > 1:
            raise Exception(f"Directory {seed_path} contains more than 1 model")


def log_args(args):
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)