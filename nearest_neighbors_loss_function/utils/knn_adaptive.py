import torch

class KNeigborsAdaptiveClassifier():

    def __init__(self, initial_n_neighbors, density_function):
        self.initial_n_neighbors = initial_n_neighbors
        self.density_function = density_function
        self.pred_probas = None

    def fit(self, distances, labels):

        n_neigbors_per_row = self._get_n_neigbors_per_row(distances)
        
        # fill diagonal with inf to ommit these elements
        distances_diag_inf = distances.masked_fill(
            torch.eye(distances.size(0), dtype=torch.bool, device=distances.device),
            float('inf')
        )

        # indices of k nearest neighbors
        _, idx = torch.topk(distances_diag_inf, k=self.initial_n_neighbors, dim=1, largest=False)

        # labels of k nearest neighbors        
        labels_k_nn = labels[idx]

        # mask to select specific number of neighbors per row
        mask = torch.arange(self.initial_n_neighbors)[None, :] < n_neigbors_per_row[:, None]
        
        self.pred_probas = (labels_k_nn * mask).sum(dim=1) / n_neigbors_per_row


    def _get_n_neigbors_per_row(self, distances):

        local_density = self._local_reachability_density(distances)
        avg_density = torch.mean(local_density)
        
        n_neigbors_per_row_computed = torch.floor(self.initial_n_neighbors * (avg_density / local_density))
        n_neigbors_per_row_min = torch.full_like(n_neigbors_per_row_computed, self.initial_n_neighbors)
        n_neigbors_per_row = torch.minimum(n_neigbors_per_row_computed, n_neigbors_per_row_min)

        return n_neigbors_per_row

    def _local_reachability_density(self, distances):
        '''
        local reachability distance calculated according to the LOF method
        
        lrd = n_neighbors / sum_reachability_distances
        '''

        # distances to n nearest neighbors
        indices_nn = torch.argsort(distances, dim=1)[:, 1:self.initial_n_neighbors+1] # skip self
        distances_nn = torch.gather(distances, 1, indices_nn)

        # distances to the most distance neighbor
        k_dist = torch.sort(distances_nn, dim=1).values[:, self.initial_n_neighbors-1]

        # k_dist for all of the k neighbors
        k_dist_neighbors = k_dist[indices_nn]

        # reachability distances necessary to calculate the local density
        reach_dist = torch.maximum(distances_nn, k_dist_neighbors)
        
        # local_reachability_density
        lrd = self.initial_n_neighbors / (torch.sum(reach_dist, dim=1) + 1e-10)
        
        return lrd
    

    def predict_proba(self, id):
        return self.pred_probas[id]