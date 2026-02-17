from .generate_embeddings import generate_embeddings
from sklearn.neighbors import KNeighborsClassifier
import torch
import math

class GammaCalculator():

    def __init__(self, embedding_length, n_neighbors, batch_size, device, gamma_function, focal_pow = 1, recalculation_strategy = 0):
        self.embedding_length = embedding_length
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.device = device
        self.gamma_values = None
        self.gamma_function = gamma_function
        self.focal_pow = focal_pow
        self.n_samples = None
        self.recalculation_strategy = recalculation_strategy


    def recalculate_gamma_values(self, model, data_loader, n_samples, batch_id):
        
        if self.recalculation_strategy == 0 and batch_id == 0:
            self._refresh_knn(model, data_loader, n_samples)

        elif self.recalculation_strategy > 0 and batch_id % self.recalculation_strategy == 0:
            self._refresh_knn(model, data_loader, n_samples)

        elif self.recalculation_strategy == -1:
            self.gamma_values = torch.ones(n_samples)
            self.n_samples = n_samples

        self.gamma_values = self.gamma_values.to(self.device)


    def _refresh_knn(self, model, data_loader, n_samples):

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
                gamma = self._calculate_gamma(sample_proba)
                gamma_values[sample_id] = gamma

        self.gamma_values = gamma_values
        self.n_samples = n_samples


    def get_gamma_values(self, gamma_start_id, gamma_end_id):
        return self.gamma_values[gamma_start_id: gamma_end_id]


    def _calculate_gamma(self, sample_proba):

        if self.gamma_function == "boosted_gamma":
            return self._boosted_gamma(sample_proba)
        elif self.gamma_function == "focal_gamma":
            return self._focal_gamma(sample_proba)
        else:
            raise Exception(f"Gamma function {self.gamma_function} not implemented")


    def _focal_gamma(self, sample_proba):
        gamma = - math.pow(1 - sample_proba, self.focal_pow) * math.log(sample_proba)
        return gamma
    

    def _boosted_gamma(self, sample_proba):
        gamma = 2 - sample_proba
        return gamma