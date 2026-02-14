from .generate_embeddings import generate_embeddings
from sklearn.neighbors import KNeighborsClassifier
import torch

class GammaCalculator():

    def __init__(self, embedding_length, n_neighbors, batch_size, device, recalculation_strategy = 0):
        self.embedding_length = embedding_length
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.device = device
        self.gamma_values = None
        self.n_samples = None
        self.recalculation_strategy = recalculation_strategy


    def recalculate_gamma_values(self, model, data_loader, n_samples, batch_id):
        
        if self.recalculation_strategy == 0 and batch_id == 0:
            X, y = generate_embeddings(model, data_loader, n_samples, self.embedding_length, self.device)
            y = y.ravel()
            proba_thrash_threshold = 1 / self.n_neighbors
            gamma_values = torch.ones(n_samples)

            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs=-1)
            knn.fit(X, y)

            for sample_id, sample_label in enumerate(y):
                if sample_label == 1:
                    sample_embedding = X[sample_id, :].reshape(1, -1)
                    sample_proba = knn.predict_proba(sample_embedding)[0][1] - proba_thrash_threshold
                    gamma = 2 - sample_proba
                    gamma_values[sample_id] = gamma

            self.gamma_values = gamma_values
            self.n_samples = n_samples

        elif self.recalculation_strategy == -1:
            self.gamma_values = torch.ones(n_samples)
            self.n_samples = n_samples

        self.gamma_values = self.gamma_values.to(self.device)

    def get_gamma_values(self, gamma_start_id, gamma_end_id):
        return self.gamma_values[gamma_start_id: gamma_end_id]