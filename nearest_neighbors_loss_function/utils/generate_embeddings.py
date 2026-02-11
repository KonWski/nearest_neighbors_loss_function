import torch

def generate_embeddings(model, data_loader, embedding_length):

    model.eval()
    embeddings = torch.zeros([data_loader.dataset.data.x.shape[0], embedding_length], dtype=float)
    labels = torch.zeros([data_loader.dataset.data.x.shape[0]], dtype=int)

    offset = 0
    for _, data in enumerate(data_loader):
        batch_embeddings = model(data.anchor_X, data.anchor_edge_index, data.anchor_batch)
        n_embeddings = batch_embeddings.shape[0]
        embeddings[offset: offset + n_embeddings] = batch_embeddings
        labels[offset: offset + n_embeddings] = data.y
        offset += 1

    return embeddings, labels