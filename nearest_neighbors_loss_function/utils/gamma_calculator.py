from .generate_embeddings import generate_embeddings
from sklearn.neighbors import KNeighborsClassifier
import torch

class GammaCalculator():

    def __init__(self, embedding_length, n_neighbors, batch_size, recalculation_strategy = 0):
        self.embedding_length = embedding_length
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.gamma_values = None
        self.n_samples = None
        self.recalculation_strategy = recalculation_strategy


    def recalculate_gamma_values(self, model, data_loader, batch_id):
        
        if self.recalculation_strategy == 0 and batch_id == 0:
            X, y = generate_embeddings(model, data_loader, self.embedding_length)
            n_samples = len(y)
            proba_thrash_threshold = 1 / self.n_neighbors
            gamma_values = torch.ones(n_samples)

            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            knn.fit(X, y)

            for sample_id, sample_label in enumerate(y):
                if sample_label == 1:
                    sample_embedding = X[sample_id, :]
                    print(f"sample_embedding.shape: {sample_embedding.shape}")
                    sample_embedding = sample_embedding.reshape(1, -1)
                    print(f"After reshape sample_embedding.shape: {sample_embedding.shape}")
                    sample_proba = knn.predict_proba(sample_embedding)[0][1]
                    sample_proba = sample_proba - proba_thrash_threshold
                    print(f"sample_proba: {sample_proba}")
                    print(f"sample_proba.shape: {sample_proba.shape}")
                    gamma = 2 - sample_proba
                    print(f"gamma: {gamma}")
                    # print(f"gamma.shape: {gamma.shape}")


                    gamma_values[sample_id] = gamma

            self.gamma_values = gamma_values
            self.n_samples = n_samples
    

    def get_gamma_values(self, batch_id):
        return self.gamma_values[batch_id: (1 + batch_id) * self.batch_size]