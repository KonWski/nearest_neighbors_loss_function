import torch
from .datasets import get_loaders
from .train import train_triplet
from .evaluate_model import test_model
import logging
from nearest_neighbors_loss_function.utils.statistics import Statistics
import os

def train_workflow(args):
    
    log_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = get_loaders(args.batch_size, debug=args.debug)

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
        n_neighbors=args.n_neighbors,
        n_epochs=args.n_epochs,
        save_path=args.save_path,
        lr=args.lr,
        model_name=args.model_name,
        model_in_channels=args.model_in_channels,
        model_hidden_channels=args.model_hidden_channels,
        embedding_length=args.embedding_length,
        optimized_param_name=args.optimized_param_name,
        device=device
    )

    statistics = test_model(args.model_name, statistics, best_seed_models, args.model_in_channels, args.model_hidden_channels, 
                            args.embedding_length, train_loader, n_train_samples, test_loader, 
                            args.n_neighbors, device)

    statistics.save()


def test_workflow(args):

    log_args(args)
    experiment_hash = args.save_path.split("/")[-1]
    statistics = Statistics(args.n_epochs, args.save_path, experiment_hash, args.optimized_param_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, test_loader = get_loaders(args.batch_size, debug=False)
    n_train_samples = len(train_loader.dataset)
    best_seed_models = {}

    for seed in os.listdir(args.save_path):

        # double check if its the seed dir
        try:
            int(seed)
        except ValueError:
            continue
        
        seed_path = os.path.join(args.save_path, str(seed))
        models = list(seed_path.rglob("*.pt"))
        n_models = len(models)

        if n_models > 1:
            raise Exception(f"Directory {seed_path} contains more than 1 model")

        model_path = os.path.join(seed_path, models[0])
        best_seed_models[seed] = {"model_path": model_path}

        statistics = test_model(args.model_name, statistics, best_seed_models, args.model_in_channels, args.model_hidden_channels, 
                                args.embedding_length, train_loader, n_train_samples, test_loader, 
                                args.n_neighbors, device)

    statistics.save()


def log_args(args):
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)