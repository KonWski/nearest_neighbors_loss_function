from .generate_embeddings import generate_embeddings
from .knn_adaptive import KNeigborsAdaptiveClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
import math
import torch.nn.functional as F

class GammaCalculator():

    def __init__(
            self, 
            embedding_length, 
            n_neighbors, 
            batch_size, 
            device, 
            gamma_function, 
            focal_pow, 
            recalculation_strategy,
            weight_distances,
            density_awareness, 
            density_function,
            samples_difficultness, 
            lambda_samples_difficultness
        ):
        self.embedding_length = embedding_length
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.device = device
        self.gamma_values = None
        self.gamma_function = gamma_function
        self.focal_pow = focal_pow
        self.n_samples = None
        self.recalculation_strategy = recalculation_strategy
        self.weight_distances = weight_distances
        self.density_awareness = density_awareness
        self.density_function = density_function
        self.samples_difficultness = samples_difficultness
        self.lambda_samples_difficultness = lambda_samples_difficultness

    def recalculate_gamma_values(self, model, data_loader, n_samples, batch_id):
        
        if self.recalculation_strategy == 0 and batch_id == 0:
            self._refresh_gamma_values(model, data_loader, n_samples)
            self.gamma_values = self.gamma_values.to(self.device)

        elif self.recalculation_strategy > 0 and batch_id % self.recalculation_strategy == 0:
            self._refresh_gamma_values(model, data_loader, n_samples)
            self.gamma_values = self.gamma_values.to(self.device)

        elif self.recalculation_strategy == -1:
            self.gamma_values = torch.ones(n_samples)
            self.n_samples = n_samples
            self.gamma_values = self.gamma_values.to(self.device)

    def _refresh_gamma_values(self, model, data_loader, n_samples):

        X, y = generate_embeddings(model, data_loader, n_samples, self.embedding_length, self.device)
        self.y = y
        X = F.normalize(X, dim=1)
        distances = torch.cdist(X,X)

        y = y.ravel()
        proba_thrash_threshold = 1 / self.n_neighbors
        self.n_samples = n_samples
        mask = (y == 1)

        if self.density_awareness:
            knn = KNeigborsAdaptiveClassifier(self.n_neighbors, self.density_function)
            knn.fit(distances, y)
            proba_1 = knn.predict_proba(mask)
            gamma_1 = self._calculate_gamma(proba_1)
            self.gamma_values = torch.ones(n_samples, dtype=gamma_1.dtype)
            self.gamma_values[mask] = gamma_1

        else:

            if self.weight_distances:
                knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs=-1, metric="precomputed", weights="distance")
            else:
                knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs=-1, metric="precomputed")

            # convert torch -> numpy
            distances = distances.numpy()
            X = X.numpy()
            y = y.numpy()

            knn.fit(distances, y)
            distances_1 = distances[mask, :]
            proba_1 = knn.predict_proba(distances_1)[0][1] - proba_thrash_threshold
            gamma_1 = self._calculate_gamma(proba_1)
            self.gamma_values = torch.ones(n_samples)
            self.gamma_values[mask] = gamma_1


    def get_gamma_values(self, gamma_start_id, gamma_end_id, positive_mf_distances, negative_mf_distances):
        batch_gamma_values = self.gamma_values[gamma_start_id: gamma_end_id]
        print(f"Gamma labels: {self.y[gamma_start_id: gamma_end_id]}")
        
        if self.density_awareness:
            samples_difficultness = positive_mf_distances / negative_mf_distances
            batch_gamma_values = batch_gamma_values * (1 + self.lambda_samples_difficultness * samples_difficultness)

        return batch_gamma_values.detach()


    def _calculate_gamma(self, sample_proba):

        if self.gamma_function == "boosted_gamma":
            return self._boosted_gamma(sample_proba)
        elif self.gamma_function == "focal_gamma":
            return self._focal_gamma(sample_proba)
        else:
            raise Exception(f"Gamma function {self.gamma_function} not implemented")


    def _focal_gamma(self, sample_proba):
        sample_proba = max(sample_proba, 1e-8)
        gamma = - math.pow(1 - sample_proba, self.focal_pow) * math.log(sample_proba)

        if gamma < 1:
            return 1

        return gamma
    

    def _boosted_gamma(self, sample_proba):
        gamma = 2 - sample_proba
        return gamma