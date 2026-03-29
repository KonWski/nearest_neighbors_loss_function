import torch

class KNeigborsAdaptiveClassifier():

  def __init__(self, initial_n_neighbors, density_function):
      self.initial_n_neighbors = initial_n_neighbors
      self.density_function = density_function
      self.probas = None

  def fit(self, distances, labels):

      n_neigbors_per_row = self._get_n_neigbors_per_row(distances)
      print(f"self.initial_n_neighbors: {self.initial_n_neighbors}")

      max_n_neigbors = int(n_neigbors_per_row.max().item())
      print(f"max_n_neigbors: {max_n_neigbors}")
      _, idx = torch.topk(distances, k=max_n_neigbors, dim=1, largest=False)

      mask = torch.arange(max_n_neigbors)[None, :] < n_neigbors_per_row[:, None]
      row_ids = torch.arange(distances.size(0))[:, None].expand_as(idx)

      selected_rows = row_ids[mask]
      selected_labels = labels[selected_rows]

      row_sums = torch.zeros(distances.size(0), dtype=selected_rows.dtype)
      row_sums.scatter_add_(0, selected_rows, selected_labels)

      self.probas = row_sums / n_neigbors_per_row

  def _get_n_neigbors_per_row(self, distances):

      density = self._local_reachability_distance(distances)
      print(f"density: {density}")
      avg_density = torch.mean(density)
      print(f"avg_density: {avg_density}")
      n_neigbors_per_row = torch.floor(self.initial_n_neighbors * (avg_density / density))
      return n_neigbors_per_row

  def _local_reachability_distance(self, distances):

      indices = torch.argsort(distances, dim=1)[:, 1:self.initial_n_neighbors+1] # skip self
      distances = torch.gather(distances, 1, indices)

      k_dist = torch.sort(distances, dim=1).values[:, self.initial_n_neighbors-1]

      k_dist_neighbors = k_dist[indices]
      reach_dist = torch.maximum(distances, k_dist_neighbors)

      lrd = 1.0 / (torch.mean(reach_dist, dim=1) + 1e-10)
      return lrd

  def predict_proba(self, id):
      return self.probas[id]