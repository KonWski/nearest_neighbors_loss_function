from nearest_neighbors_loss_function.utils.auxiliary_functions import create_experiment_dir, \
    create_model_dir, set_seed, adjust_graph_data_dtype, get_model
from nearest_neighbors_loss_function.utils.gamma_calculator import GammaCalculator
from nearest_neighbors_loss_function.utils.statistics import Statistics
from nearest_neighbors_loss_function.utils.evaluate_model import evaluate_model
from nearest_neighbors_loss_function.utils.checkpoints import save_model
from uuid import uuid4
from torch.optim import Adam
import logging
import torch
from typing import List
import numpy as np
import os
from .generate_embeddings import generate_embeddings
from torchvision.ops import sigmoid_focal_loss
from nearest_neighbors_loss_function.utils.focal_batch_shaper import FocalBatchShaper

def train_focal(
        seeds: List[int], 
        train_loader, 
        valid_loader, 
        batch_size: int, 
        weight_distances: bool,
        density_awareness: bool,
        density_function: str,
        samples_difficultness: bool,
        lambda_samples_difficultness: float,
        n_evaluation_models: int,
        n_neighbors: int, 
        n_epochs: int, 
        save_path: str, 
        lr: float, 
        model_name: str,
        model_in_channels: int,
        model_hidden_channels: int,
        model_n_blocks: int,  
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

        logging.info(f"Running training process for seed: {seed}. Progress: {id_seed + 1}/{len(seeds)}")
        model_dir_path = create_model_dir(experiment_dir_path, seed)
        set_seed(seed)

        batch_shaper = FocalBatchShaper(device)
        model = get_model(model_name, model_in_channels, model_hidden_channels, model_n_blocks, embedding_length)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)

        # save auxiliary params
        best_scores = []
        best_epochs = []
        best_model_paths = []

        for epoch in range(0, n_epochs):

            model_epoch_hash = uuid4().hex
            logging.info(f"Epoch: {epoch + 1}/{n_epochs}")

            model, optimizer, train_basic_stats = train(model, train_loader, batch_shaper, optimizer, seed, epoch, model_epoch_hash, device)

            train_embeddings, train_labels = generate_embeddings(model, train_loader, n_train_samples, embedding_length, device)
            valid_embeddings, valid_labels = generate_embeddings(model, valid_loader, n_valid_samples, embedding_length, device)

            train_stats, embeddings_with_nans = evaluate_model(model, "train", train_embeddings, train_labels, train_embeddings, train_labels, n_neighbors, "train")
            valid_stats, embeddings_with_nans = evaluate_model(model, "valid", train_embeddings, train_labels, valid_embeddings, valid_labels, n_neighbors, "valid")

            # early exit
            if embeddings_with_nans:
                logging.info(f"Embeddings generated during model evaluation contained nans -> next split")
                break
            elif valid_stats["valid_precision"] == 0.0:
                logging.info(f"Precision 0 reached at epoch {epoch} -> next split")
                break
            
            train_stats = train_basic_stats | train_stats
            statistics.add(train_stats, valid_stats)
            statistics.log_last_train_stats()

            valid_optimized_param_value = valid_stats[f"valid_{optimized_param_name}"]
            train_optimized_param_value = train_stats[f"train_{optimized_param_name}"]
            penalty = max(train_optimized_param_value - valid_optimized_param_value, 0)            
            score = valid_optimized_param_value - 0.25 * penalty
            n_best_models = len(best_scores) 
            
            if score > min(best_scores, default=float("-inf")) or n_best_models < n_evaluation_models:

                best_model_path = save_model(model_dir_path, experiment_hash, seed, epoch, model_epoch_hash, lr, model.state_dict(), 
                        train_stats["loss"], n_neighbors, valid_optimized_param_value, "", batch_size, 
                        "", density_awareness, samples_difficultness, lambda_samples_difficultness)

                if n_best_models < n_evaluation_models:
                    best_scores.append(valid_optimized_param_value)
                    best_model_paths.append(best_model_path)
                    best_epochs.append(epoch)

                # replace worst with best model
                elif n_best_models == n_evaluation_models:
                    
                    id_worst_model = np.argmin(best_scores)
                    os.remove(best_model_paths[id_worst_model])

                    best_scores[id_worst_model] = valid_optimized_param_value
                    best_model_paths[id_worst_model] = best_model_path
                    best_epochs[id_worst_model] = epoch
            
            if statistics.early_stop_training(seed):
                logging.info(f"Training stopped because of lack of improvement. Current window mean: {statistics.current_window_mean}, last window mean: {statistics.last_window_mean}")
                break

        best_seed_models[seed] = {"epoch": best_epochs, "model_path": best_model_paths}

    return statistics, best_seed_models, n_train_samples


def train(model, train_loader, batch_shaper, optimizer, seed, epoch, model_epoch_hash, device):

    model.train()

    # calculated parameters
    running_loss = 0.0

    # auxiliary param for selecting gamma values
    gamma_start_id = 0

    for data_id, data in enumerate(train_loader):

        data = data.to(device)
        labels = data.y
        n_samples = labels.shape[0]

        with torch.set_grad_enabled(True):
            
            model.train()
            optimizer.zero_grad()
            data = adjust_graph_data_dtype(data, model)
            anchor_mfs = model(data)

            anchors_l, anchors_r, labels = batch_shaper.shape_batch(anchor_mfs, labels)
            logits = model.classify(anchors_l, anchors_r)
            loss = sigmoid_focal_loss(logits, labels).mean()

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            gamma_start_id += n_samples

    epoch_loss = round(running_loss / (data_id + 1), 5)
    train_stats = {"seed": seed, "epoch": epoch, "loss": epoch_loss, "model_epoch_hash": model_epoch_hash}

    return model, optimizer, train_stats