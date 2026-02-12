import torch

def generate_embeddings(model, data_loader, embedding_length, batch_size):

    model.eval()
    try:
       original_shuffle = data_loader.batch_sampler.shuffle
    except AttributeError: 
        original_shuffle = False

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = False

    embeddings = torch.zeros([data_loader.dataset.data.x.shape[0], embedding_length], dtype=float)
    labels = torch.zeros([data_loader.dataset.data.x.shape[0], 1], dtype=int)

    offset = 0
    for _, data in enumerate(data_loader):

        batch_embeddings = model(data.x.float(), data.edge_index, data.batch)
        embeddings[offset * batch_size: (offset + 1) * batch_size] = batch_embeddings
        labels[offset * batch_size: (offset + 1) * batch_size] = data.y
        offset += 1

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = True

    embeddings = embeddings.detach().numpy()
    labels = labels.detach().numpy()

    return embeddings, labels