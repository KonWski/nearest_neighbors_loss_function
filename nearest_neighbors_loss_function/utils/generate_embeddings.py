import torch

def generate_embeddings(model, data_loader, n_samples, embedding_length):

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
    for _, data in enumerate(data_loader):
        
        if _ <= 1:
            print(8*"-")
            print(f"Generating embeddings x and y elements")
            print(data.x.float())
            print(data.x.float().shape)
            print(data.y.T)
            print(8*"-")

        n_samples_batch = data.y.shape[0]
        batch_embeddings = model(data.x.float(), data.edge_index, data.batch)
        embeddings[start_id: start_id + n_samples_batch] = batch_embeddings
        labels[start_id: start_id + n_samples_batch] = data.y
        start_id += n_samples_batch

    if original_shuffle == True:
        data_loader.batch_sampler.shuffle = True

    embeddings = embeddings.detach().numpy()
    labels = labels.detach().numpy()

    return embeddings, labels