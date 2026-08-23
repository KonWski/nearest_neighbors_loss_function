import torch
from nearest_neighbors_loss_function.utils.transformations import FloatTransformation
from nearest_neighbors_loss_function.utils.auxiliary_functions import set_seed
from .checkpoints import load_model
import numpy as np

def generate_embeddings(model, data_loader, n_samples, embedding_length, use_original_transformation, device):

    model.eval()
    try:
        original_shuffle = data_loader.batch_sampler.shuffle
    except AttributeError: 
        original_shuffle = False

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = False

    original_transform = data_loader.dataset.transform

    if not use_original_transformation:
        data_loader.dataset.transform = FloatTransformation()

    embeddings = torch.zeros([n_samples, embedding_length + data_loader.dataset.embedding_extra_length], dtype=float)
    labels = torch.zeros([n_samples, 1], dtype=int)

    start_id = 0
    with torch.no_grad():
        
        for _, data in enumerate(data_loader):
            print(f"data id: {_}") 
            data = data.to(device)
            n_samples_batch = data.y.shape[0]
            model_embeddings = model(data)
            batch_embeddings = [model_embeddings]
            print(f"batch_embeddings.shape: {model_embeddings.shape}")

            # concatenate fingegrprints from the chosen methods
            if data_loader.dataset.use_extra_embeddings:
                for embedding_name in data_loader.dataset.extra_embeddings_methods:
                    print(f"embedding_name: {embedding_name}")
                    print(f"type(getattr(data, embedding_name)): {type(getattr(data, embedding_name))}")
                    print(getattr(data, embedding_name)[:2])
                    print(getattr(data, embedding_name)[0].shape)
                    print(getattr(data, embedding_name)[0].dtype)
                    print(f"len(getattr(data, embedding_name)): {len(getattr(data, embedding_name))}")
                    embeddings = torch.stack([torch.from_numpy(x) for x in getattr(data, embedding_name)])
                    embeddings = embeddings.to(device)
                    print("Stacked tensors")
                    batch_embeddings.append(embeddings)
                
                # extra_fingerprints = [getattr(data, embedding_name) 
                #                     for embedding_name in data_loader.dataset.extra_embeddings_methods]
                batch_embeddings = torch.concat(batch_embeddings, axis=1)
                print("Concatenated embeddings")

            embeddings[start_id: start_id + n_samples_batch] = batch_embeddings.detach().cpu()
            labels[start_id: start_id + n_samples_batch] = data.y
            start_id += n_samples_batch

            del data, model_embeddings

    # return to initial settings
    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = True

    data_loader.dataset.transform = original_transform

    embeddings = embeddings.detach()
    labels = labels.detach()

    return embeddings, labels


def generate_all_splits_embeddings(seed, train_loader, n_train_samples, valid_loader, n_valid_samples, test_loader, 
                                   n_test_samples, model_path, model_name, model_in_channels, model_hidden_channels, 
                                   model_n_blocks, embedding_length, device):
    
    set_seed(seed)
    train_loader.batch_sampler.shuffle_data()
    model, _ = load_model(model_path, model_name, model_in_channels, model_hidden_channels, 
                                   model_n_blocks, embedding_length)
    model.to(device)

    train_embeddings, train_labels = generate_embeddings(model, train_loader, n_train_samples, embedding_length, False, device)
    valid_embeddings, valid_labels = generate_embeddings(model, valid_loader, n_valid_samples, embedding_length, False, device)        
    test_embeddings, test_labels = generate_embeddings(model, test_loader, n_test_samples, embedding_length, False, device)

    return train_embeddings, train_labels, valid_embeddings, valid_labels, test_embeddings, test_labels