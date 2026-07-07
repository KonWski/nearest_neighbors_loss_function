import torch
from .auxiliary_functions import adjust_graph_data_dtype

def generate_embeddings(model, data_loader, n_samples, embedding_length, device):

    model.eval()
    try:
        original_shuffle = data_loader.batch_sampler.shuffle
    except AttributeError: 
        original_shuffle = False

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = False

    embeddings = torch.zeros([n_samples, embedding_length], dtype=float)
    labels = torch.zeros([n_samples, 1], dtype=int)

    start_id = 0
    with torch.no_grad():
        
        for _, data in enumerate(data_loader):

            data = data.to(device)
            data = adjust_graph_data_dtype(data, model)
            n_samples_batch = data.y.shape[0]
            batch_embeddings = model(data)
            embeddings[start_id: start_id + n_samples_batch] = batch_embeddings.detach().cpu()
            labels[start_id: start_id + n_samples_batch] = data.y
            start_id += n_samples_batch

            del data, batch_embeddings

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = True

    embeddings = embeddings.detach()
    labels = labels.detach()

    return embeddings, labels