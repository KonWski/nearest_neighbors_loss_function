from nearest_neighbors_loss_function.utils.batch_shaper import BatchShaper
from nearest_neighbors_loss_function.utils.auxiliary_functions import create_experiment_dir, \
    create_model_dir, set_seed, adjust_graph_data_dtype, get_model
from nearest_neighbors_loss_function.utils.gamma_calculator import GammaCalculator
from nearest_neighbors_loss_function.utils.statistics import Statistics
from nearest_neighbors_loss_function.utils.evaluate_model import evaluate_model
from nearest_neighbors_loss_function.utils.checkpoints import save_model
from uuid import uuid4
from torch.nn import TripletMarginLoss
from torch.optim import Adam
import logging
import torch
from typing import List
import numpy as np
import os

def train_triplet(
        seeds: List[int], 
        train_loader, 
        valid_loader, 
        training_type: str, 
        batch_size: int, 
        gamma_recalculation_strategy: int, 
        gamma_function: str,
        weight_distances: bool,
        focal_pow: float,
        density_awareness: bool,
        density_function: str,
        samples_difficultness: bool,
        lambda_samples_difficultness: float,
        n_evaluation_models: int,
        evaluation_model_name: str,
        n_neighbors: int, 
        n_epochs: int, 
        save_path: str, 
        lr: float, 
        model_name: str,
        model_in_channels: int,
        model_hidden_channels: int, 
        embedding_length: int, 
        optimized_param_name: str,
        early_stop_window_size: int,
        device
    ):

    logging.info(f"Initiating experiment")

    experiment_hash = uuid4().hex
    experiment_dir_path = create_experiment_dir(save_path, experiment_hash)
    
    statistics = Statistics(n_epochs, experiment_dir_path, experiment_hash, optimized_param_name, early_stop_window_size)
    best_seed_models = {}

    n_train_samples = len(train_loader.dataset)
    n_valid_samples = len(valid_loader.dataset)

    # iterate over all given seeds
    for id_seed, seed in enumerate(seeds):

        loss_function = TripletMarginLoss(reduction="none", margin=0.5)
        batch_shaper = BatchShaper(device, training_type)
        gamma_calculator = GammaCalculator(embedding_length, n_neighbors, batch_size, device, gamma_function, focal_pow, 
                                           gamma_recalculation_strategy, weight_distances, density_awareness, density_function,
                                           samples_difficultness, lambda_samples_difficultness)

        logging.info(f"Running training process for seed: {seed}. Progress: {id_seed + 1}/{len(seeds)}")
        model_dir_path = create_model_dir(experiment_dir_path, seed)
        set_seed(seed)

        model = get_model(model_name, model_in_channels, model_hidden_channels, embedding_length)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)

        # save auxiliary params
        best_optimized_param_values = []
        best_epochs = []
        best_model_paths = []

        for epoch in range(0, n_epochs):

            model_epoch_hash = uuid4().hex
            logging.info(f"Epoch: {epoch + 1}/{n_epochs}")

            model, optimizer, loss_function, train_stats = train(model, train_loader, n_train_samples, optimizer, loss_function, 
                                                                 batch_shaper, gamma_calculator, seed, epoch, model_epoch_hash, device)

            validate_stats, embeddings_with_nans = evaluate_model(model, evaluation_model_name, train_loader, n_train_samples, valid_loader, n_valid_samples, embedding_length, 
                                        n_neighbors, "valid", device)

            # early exit
            if embeddings_with_nans:
                logging.info(f"Embeddings generated during model evaluation contained nans -> next split")
                break
            elif validate_stats["valid_precision"] == 0.0:
                logging.info(f"Precision 0 reached at epoch {epoch} -> next split")
                break

            statistics.add(train_stats, validate_stats)
            statistics.log_last_train_stats()

            optimized_param_value = validate_stats[f"valid_f1"]
            n_best_models = len(best_optimized_param_values) 
            
            if optimized_param_value > min(best_optimized_param_values, default=float("-inf")) or n_best_models < n_evaluation_models:

                best_model_path = save_model(model_dir_path, experiment_hash, seed, epoch, model_epoch_hash, lr, model.state_dict(), 
                        train_stats["loss"], n_neighbors, optimized_param_value, training_type, batch_size, 
                        gamma_recalculation_strategy, density_awareness, samples_difficultness, lambda_samples_difficultness)

                if n_best_models < n_evaluation_models:
                    best_optimized_param_values.append(optimized_param_value)
                    best_model_paths.append(best_model_path)
                    best_epochs.append(epoch)

                # replace worst with best model
                elif n_best_models == n_evaluation_models:
                    
                    id_worst_model = np.argmin(best_optimized_param_values)
                    os.remove(best_model_paths[id_worst_model])

                    best_optimized_param_values[id_worst_model] = optimized_param_value
                    best_model_paths[id_worst_model] = best_model_path
                    best_epochs[id_worst_model] = epoch
            
            if statistics.early_stop_training(seed):
                logging.info(f"Training stopped because of lack of improvement. Current window mean: {statistics.current_window_mean}, last window mean: {statistics.last_window_mean}")
                break

        best_seed_models[seed] = {"epoch": best_epochs, "model_path": best_model_paths}

    return statistics, best_seed_models, n_train_samples


def train(model, train_loader, n_train_samples, optimizer, loss_function, batch_shaper, gamma_calculator, seed, epoch, model_epoch_hash, device):

    model.train()

    # calculated parameters
    running_loss = 0.0

    # auxiliary param for selecting gamma values
    gamma_start_id = 0

    for data_id, data in enumerate(train_loader):

        data = data.to(device)
        gamma_calculator.recalculate_gamma_values(model, train_loader, n_train_samples, data_id)
        labels = data.y
        n_samples = labels.shape[0]
        gamma_end_id = gamma_start_id + n_samples

        with torch.set_grad_enabled(True):
            
            model.train()
            optimizer.zero_grad()
            data = adjust_graph_data_dtype(data, model)
            anchor_mfs = model(data)
            anchor_mf, positive_mf, positive_mf_distances, negative_mf, negative_mf_distances, _ = batch_shaper.shape_batch(anchor_mfs, labels)

            loss = loss_function(anchor_mf, positive_mf, negative_mf)
            gamma_values = gamma_calculator.get_gamma_values(gamma_start_id, gamma_end_id, positive_mf_distances, negative_mf_distances)
            gamma_values = gamma_values.detach()
            gamma_values = gamma_values.view_as(loss)
            loss = loss * gamma_values
            loss = loss.mean()

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            gamma_start_id += n_samples

    epoch_loss = round(running_loss / (data_id + 1), 5)
    train_stats = {"seed": seed, "epoch": epoch, "loss": epoch_loss, "model_epoch_hash": model_epoch_hash}

    return model, optimizer, loss_function, train_stats