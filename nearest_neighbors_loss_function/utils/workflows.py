import torch
from .datasets import get_loaders
from .train import train_triplet
from .evaluate_model import test_model
import logging
from nearest_neighbors_loss_function.utils.statistics import Statistics
from nearest_neighbors_loss_function.utils.auxiliary_functions import set_seed
import os
from pathlib import Path

def train_workflow(args):
    
    log_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = get_loaders(args.batch_size, args.dataset_name, debug=args.debug)

    statistics, best_seed_models, n_train_samples = train_triplet(
        seeds=args.seeds,
        train_loader=train_loader,
        valid_loader=valid_loader,
        training_type=args.training_type,
        batch_size=args.batch_size,
        gamma_recalculation_strategy=args.gamma_recalculation_strategy,
        gamma_function=args.gamma_function,
        weight_distances=args.weight_distances,
        focal_pow=args.focal_pow,
        density_awareness=args.density_awareness,
        density_function=args.density_function,
        samples_difficultness=args.samples_difficultness,
        lambda_samples_difficultness=args.lambda_samples_difficultness,
        n_evaluation_models=args.n_evaluation_models,
        evaluation_model_name=args.evaluation_model_name,
        n_neighbors=args.n_neighbors,
        n_epochs=args.n_epochs,
        save_path=args.save_path,
        lr=args.lr,
        model_name=args.model_name,
        model_in_channels=args.model_in_channels,
        model_hidden_channels=args.model_hidden_channels,
        embedding_length=args.embedding_length,
        optimized_param_name=args.optimized_param_name,
        early_stop_window_size=args.early_stop_window_size,
        device=device
    )

    statistics = test_model(
        args.model_name,
        args.evaluation_model_name, 
        statistics, 
        best_seed_models, 
        args.model_in_channels, 
        args.model_hidden_channels, 
        args.embedding_length, 
        train_loader,
        n_train_samples,
        test_loader,
        args.n_neighbors,
        "test", 
        device
    )

    statistics.save()


def test_workflow(args):

    log_args(args)
    experiment_hash = args.save_path.split("/")[-1]
    statistics = Statistics(args.n_epochs, args.save_path, experiment_hash, args.optimized_param_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = get_loaders(args.batch_size, args.dataset_name, debug=False)
    train_loader.batch_sampler.shuffle_data()
    n_train_samples = len(train_loader.dataset)
    best_seed_models = {}

    for seed in os.listdir(args.save_path):

        # double check if its the seed dir
        try:
            int(seed)
        except ValueError:
            continue
        
        set_seed(int(seed))
        seed_path = Path(os.path.join(args.save_path, str(seed)))
        models = list(seed_path.rglob("*.pt"))
        n_models = len(models)

        if n_models > 1:
            raise Exception(f"Directory {seed_path} contains more than 1 model")

        best_seed_models[seed] = {"model_path": models[0]}

    statistics = test_model(args.model_name, args.evaluation_model_name, statistics, best_seed_models, 
                            args.model_in_channels, args.model_hidden_channels, args.embedding_length, train_loader, n_train_samples, valid_loader, 
                            args.n_neighbors, "valid", device)

    statistics = test_model(args.model_name, args.evaluation_model_name, statistics, best_seed_models, 
                            args.model_in_channels, args.model_hidden_channels, args.embedding_length, train_loader, n_train_samples, test_loader, 
                            args.n_neighbors, "test", device)

    statistics.save()


def log_args(args):
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)