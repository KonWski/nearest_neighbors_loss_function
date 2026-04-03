from torch import save, load
import logging
from datetime import datetime
import os
from nearest_neighbors_loss_function.utils.auxiliary_functions import get_model

def save_model(model_dir_path, experiment_hash, seed, epoch, model_epoch_hash,  lr, model_state_dict,
               train_loss, n_neighbors, optimized_param_value, training_type, batch_size, 
               gamma_recalculation_strategy, 
               density_awareness: bool, 
               samples_difficultness: bool, 
               lambda_samples_difficultness: float):

    checkpoint = {}

    # save model to checkpoint
    checkpoint["epoch"] = epoch
    checkpoint["lr"] = lr
    checkpoint["experiment_hash"] = experiment_hash
    checkpoint["seed"] = seed
    checkpoint["model_state_dict"] = model_state_dict
    checkpoint["model_epoch_hash"] = model_epoch_hash
    checkpoint['train_loss'] = train_loss
    checkpoint["n_neighbors"] = n_neighbors
    checkpoint["optimized_param_value"] = optimized_param_value
    checkpoint["training_type"] = training_type
    checkpoint["batch_size"] = batch_size
    checkpoint["gamma_recalculation_strategy"] = gamma_recalculation_strategy
    checkpoint["density_awareness"] = density_awareness
    checkpoint["samples_difficultness"] = samples_difficultness
    checkpoint["lambda_samples_difficultness"] = lambda_samples_difficultness
    checkpoint["save_model_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    checkpoint_path = os.path.join(model_dir_path, f"{model_epoch_hash}.pt")
    save(checkpoint, checkpoint_path)
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")

    return checkpoint_path


def load_model(model_path, model_name, in_channels, hidden_dim, embedding_size):

    logging.info(f"Loading model from path: {model_path}")
    checkpoint = load(model_path, weights_only=False)
    model = get_model(model_name, in_channels, hidden_dim, embedding_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint