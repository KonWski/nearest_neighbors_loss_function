import torch

def generate_embeddings(model, data_loader, embedding_length, batch_size):

    model.eval()
    try:
       original_shuffle = data_loader.batch_sampler.shuffle
    except AttributeError: 
        original_shuffle = False

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = False

    n_samples = data_loader.dataset.data.y.shape[0]
    embeddings = torch.zeros([n_samples, embedding_length], dtype=float)
    labels = torch.zeros([n_samples, 1], dtype=int)
    print(f"labels.shape: {labels.shape}")

    start_id = 0
    for _, data in enumerate(data_loader):
        
        n_samples_batch = data.y.shape[0]
        print(f"n_samples_batch: {n_samples_batch}")
        batch_embeddings = model(data.x.float(), data.edge_index, data.batch)
        embeddings[start_id: start_id + n_samples_batch] = batch_embeddings
        labels[start_id: start_id + n_samples_batch] = data.y
        start_id += n_samples_batch

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = True

    embeddings = embeddings.detach().numpy()
    labels = labels.detach().numpy()

    return embeddings, labels