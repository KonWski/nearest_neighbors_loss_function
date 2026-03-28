import torch
import numpy as np

class KNeigborsAdaptiveClassifier():

        def __init__(self, initial_n_neighbors, density_function):
            self.initial_n_neighbors = initial_n_neighbors
            self.density_function = density_function
            self.probas = None

        def fit(self, distances, labels):

            n_neigbors_per_row = self._get_n_neigbors_per_row(distances)

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

        def _get_n_neigbors_per_row(self, distances):

            density = self._local_reachability_distance(distances)
            avg_density = np.average(density)
            n_neigbors_per_row = self.initial_n_neighbors * (avg_density / density)
            return n_neigbors_per_row

        def _local_reachability_distance(self, distances):

            indices = np.argsort(distances, axis=1)[:, 1: self.initial_n_neighbors+1]  # skip self
            distances = np.take_along_axis(distances, indices, axis=1)

            k_dist = np.sort(distances, axis=1)[:, self.initial_n_neighbors-1]

            k_dist_neighbors = k_dist[indices]
            reach_dist = np.maximum(distances, k_dist_neighbors)

            lrd = 1.0 / (np.mean(reach_dist, axis=1) + 1e-10)
            return lrd

        def predict_proba(self, id):
            return self.probas[id]