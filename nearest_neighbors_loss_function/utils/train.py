from nearest_neighbors_loss_function.utils.models import GNNModel
from nearest_neighbors_loss_function.utils.batch_shaper import BatchShaper
from nearest_neighbors_loss_function.utils.auxiliary_functions import create_experiment_dir, \
    create_model_dir, set_seed, convert_graph_data_to_float
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


def train_triplet(
        seeds: List[int], 
        train_loader, 
        valid_loader, 
        training_type: str, 
        batch_size: int, 
        gamma_recalculation_strategy: int, 
        gamma_function: str,
        density_awareness: bool,
        samples_difficultness: bool,
        lambda_samples_difficultness: float,
        n_neighbors: int, 
        n_epochs: int, 
        save_path: str, 
        lr: float, 
        model_in_channels: int,
        model_hidden_channels: int, 
        embedding_length: int, 
        optimized_param_name: str, 
        device
    ):

    logging.info(f"Initiating experiment")

    experiment_hash = uuid4().hex
    experiment_dir_path = create_experiment_dir(save_path, experiment_hash)
    
    statistics = Statistics(n_epochs, experiment_dir_path, experiment_hash, optimized_param_name)

    n_train_samples = len(train_loader.dataset)
    n_valid_samples = len(valid_loader.dataset)

    # iterate over all given seeds
    for id_seed, seed in enumerate(seeds):

        loss_function = TripletMarginLoss(reduction="none")
        batch_shaper = BatchShaper(training_type)
        gamma_calculator = GammaCalculator(embedding_length, n_neighbors, batch_size, device, gamma_function, recalculation_strategy = gamma_recalculation_strategy)

        logging.info(f"Running training process for seed: {seed}. Progress: {id_seed + 1}/{len(seeds)}")
        model_dir_path = create_model_dir(experiment_dir_path, seed)
        set_seed(seed)

        model = GNNModel(model_in_channels, model_hidden_channels, embedding_length).to(device)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        batch_shaper = BatchShaper(training_type)

        max_epoch_optimized_param_value = float("-inf")

        for epoch in range(0, n_epochs):

            logging.info(f"Epoch: {epoch + 1}/{n_epochs}")
            model_epoch_hash = uuid4().hex

            model, optimizer, loss_function, train_stats = train(model, train_loader, n_train_samples, optimizer, loss_function, 
                                                                 batch_shaper, gamma_calculator, epoch, model_epoch_hash, device)

            validate_stats = evaluate_model(model, train_loader, n_train_samples, valid_loader, n_valid_samples, embedding_length, 
                                        n_neighbors, "valid", device)

            # early exit
            if validate_stats["precision"] == 0.0:
                print(f"Precision 0 reached at epoch {epoch} -> next split")
                break

            statistics.add(train_stats, validate_stats)
            statistics.log_last_train_stats()

            if validate_stats[optimized_param_name] > max_epoch_optimized_param_value:
                max_epoch_optimized_param_value = validate_stats[optimized_param_name]
                best_model_dir_path = model_dir_path

                save_model(model_dir_path, experiment_hash, seed, epoch, model_epoch_hash, lr, model.state_dict(), 
                           train_stats["loss"], n_neighbors, max_epoch_optimized_param_value, training_type, batch_size, 
                           gamma_recalculation_strategy, density_awareness, samples_difficultness, lambda_samples_difficultness)

    return statistics, best_model_dir_path, n_train_samples


def train(model, train_loader, n_train_samples, optimizer, loss_function, batch_shaper, gamma_calculator, epoch, model_epoch_hash, device):

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

            optimizer.zero_grad()
            data = convert_graph_data_to_float(data)
            anchor_mfs = model(data)
            anchor_mf, positive_mf, negative_mf, _ = batch_shaper.shape_batch(anchor_mfs, labels)

            loss = loss_function(anchor_mf, positive_mf, negative_mf)
            gamma_values = gamma_calculator.get_gamma_values(gamma_start_id, gamma_end_id)
            loss = loss * gamma_values
            loss = loss.mean()

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            gamma_start_id += n_samples

    epoch_loss = round(running_loss / (data_id + 1), 5)
    train_stats = {"epoch": epoch, "loss": epoch_loss, "model_epoch_hash": model_epoch_hash}

    return model, optimizer, loss_function, train_stats