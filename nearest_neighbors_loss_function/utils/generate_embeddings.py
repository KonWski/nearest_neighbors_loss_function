import torch

def generate_embeddings(model, data_loader, embedding_length, batch_size):

    model.eval()
    try:
       original_shuffle = data_loader.batch_sampler.shuffle
    except AttributeError: 
        original_shuffle = False

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = False

    n_samples = data_loader.dataset.data.x.shape[0]
    embeddings = torch.zeros([n_samples, embedding_length], dtype=float)
    labels = torch.zeros([n_samples, 1], dtype=int)

    offset = 0
    for _, data in enumerate(data_loader):

        batch_embeddings = model(data.x.float(), data.edge_index, data.batch)
        embeddings[offset * batch_size: min((offset + 1) * batch_size, n_samples)] = batch_embeddings
        labels[offset * batch_size: min((offset + 1) * batch_size, n_samples)] = data.y
        offset += 1

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = True

    embeddings = embeddings.detach().numpy()
    labels = labels.detach().numpy()

    return embeddings, labels