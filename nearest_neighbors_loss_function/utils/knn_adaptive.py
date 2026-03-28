import torch
from sklearn.neighbors import KernelDensity
import numpy as np

class KNeigborsAdaptiveClassifier():

        def __init__(self, initial_n_neighbors, density_function):
            self.initial_n_neighbors = initial_n_neighbors
            self.density_function = density_function
            self.probas = None

        def fit(self, X, distances, labels):

            n_neigbors_per_row = self.get_n_neigbors_per_row(X)
            print("Got n neighbors per row")
            max_n_neigbors = n_neigbors_per_row.max()
            _, idx = torch.topk(distances, k=max_n_neigbors, dim=1, largest=False)
            print("Got torch.topk")

            mask = torch.arange(max_n_neigbors)[None, :] < n_neigbors_per_row[:, None]
            row_ids = torch.arange(distances.size(0))[:, None].expand_as(idx)

            selected_rows = row_ids[mask]
            selected_cols = idx[mask]

            selected_y = labels[selected_rows, selected_cols]

            row_sums = torch.zeros(distances.size(0), dtype=torch.float)
            row_sums.scatter_add_(0, selected_rows, selected_y.float())
            print("scatter_add_")

            probas = row_sums / n_neigbors_per_row

            self.probas = probas

        def get_n_neigbors_per_row(self, X):
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(X)
            print("kde fit")

            log_density = kde.score_samples(X)
            print("kde score_samples")

            avg_density = np.average(log_density)
            n_neigbors_per_row = self.initial_n_neighbors * (avg_density / log_density)
            return n_neigbors_per_row

        def predict_proba(self, id):
            return self.probas[id]