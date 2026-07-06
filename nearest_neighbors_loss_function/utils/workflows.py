import torch
from .datasets import get_loaders
from .train import train_triplet
from .evaluate_model import test_best_seed_model, test_model
import logging
from nearest_neighbors_loss_function.utils.statistics import Statistics
from nearest_neighbors_loss_function.utils.evaluate_model_params import EvaluateModelParams
import os
from pathlib import Path

def train_workflow(args):
    
    log_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = get_loaders(args.batch_size, args.dataset_name, args.task_id, debug=args.debug)
    evaluate_model_params = EvaluateModelParams(
        evaluation_model_name="knn",
        knn_n_neighbors=args.knn_n_neighbors
    )

    statistics, best_seed_models, n_train_samples = train_triplet(
        seeds=args.seeds,
        train_loader=train_loader,
        valid_loader=valid_loader,
        training_type=args.training_type,
        batch_size=args.batch_size,
        task_id=args.task_id,
        triplet_loss_margin=args.triplet_loss_margin,
        batch_shaper_margin=args.batch_shaper_margin,
        gamma_recalculation_strategy=args.gamma_recalculation_strategy,
        gamma_function=args.gamma_function,
        weight_distances=args.weight_distances,
        focal_pow=args.focal_pow,
        density_awareness=args.density_awareness,
        density_function=args.density_function,
        samples_difficultness=args.samples_difficultness,
        lambda_samples_difficultness=args.lambda_samples_difficultness,
        n_evaluation_models=args.n_evaluation_models,
        evaluate_model_params=evaluate_model_params,
        n_epochs=args.n_epochs,
        save_path=args.save_path,
        lr=args.lr,
        model_name=args.model_name,
        model_in_channels=args.model_in_channels,
        model_hidden_channels=args.model_hidden_channels,
        model_n_blocks=args.model_n_blocks,
        embedding_length=args.embedding_length,
        optimized_param_name=args.optimized_param_name,
        early_stop_window_size=args.early_stop_window_size,
        device=device
    )

    statistics = test_best_seed_model(
        args.model_name,
        statistics, 
        best_seed_models, 
        args.model_in_channels, 
        args.model_hidden_channels,
        args.model_n_blocks,
        args.embedding_length, 
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
    experiment_hash = args.save_path.split("/")[-1]
    statistics = Statistics(args.n_epochs, args.save_path, experiment_hash, args.optimized_param_name, -1, "test_report")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = get_loaders(args.batch_size, args.dataset_name, args.task_id, debug=False)
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


def log_args(args):
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)