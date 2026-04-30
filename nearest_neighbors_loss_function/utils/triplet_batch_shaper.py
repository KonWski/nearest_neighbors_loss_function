import torch

class TripletBatchShaper():

    def __init__(self, device, training_type = "hard_batch_learning", margin = 1.0):
        self.training_type = training_type
        self.margin = margin
        self.device = device

    def shape_batch(self, anchor_mfs, anchor_labels):

        indices_1 = (anchor_labels == 1).nonzero()[:,0].tolist()
        indices_0 = (anchor_labels == 0).nonzero()[:,0].tolist()
        
        anchor_mfs_1 = anchor_mfs[indices_1,:]
        anchor_mfs_0 = anchor_mfs[indices_0,:]
        distances = torch.cdist(anchor_mfs, anchor_mfs)
        n_anchors = anchor_mfs.shape[0]
        embedding_size = anchor_mfs.shape[1]

        positive_mfs = torch.zeros(n_anchors, embedding_size, device=self.device)
        negative_mfs = torch.zeros(n_anchors, embedding_size, device=self.device)
        positive_mfs_distances = torch.zeros(n_anchors, device=self.device)
        negative_mfs_distances = torch.zeros(n_anchors, device=self.device)

        if self.training_type == "hard_batch_learning":

            for anchor_iter in range(len(anchor_mfs)):

                anchor_label = anchor_labels[anchor_iter]

                if anchor_label == 0:
                    distances_pos = distances[anchor_iter, indices_0]
                    distances_neg = distances[anchor_iter, indices_1]
                    id_distance_pos_max = distances_pos.argmax().item()
                    id_distance_neg_min = distances_neg.argmin().item()
                    positive_mfs[anchor_iter] = anchor_mfs_0[id_distance_pos_max, :]
                    negative_mfs[anchor_iter] = anchor_mfs_1[id_distance_neg_min, :]

                elif anchor_label == 1:
                    distances_pos = distances[anchor_iter, indices_1]
                    distances_neg = distances[anchor_iter, indices_0]
                    id_distance_pos_max = distances_pos.argmax().item()
                    id_distance_neg_min = distances_neg.argmin().item()
                    positive_mfs[anchor_iter] = anchor_mfs_1[id_distance_pos_max, :]
                    negative_mfs[anchor_iter] = anchor_mfs_0[id_distance_neg_min, :]

                positive_mfs_distances[anchor_iter] = distances_pos[id_distance_pos_max]
                negative_mfs_distances[anchor_iter] = distances_pos[id_distance_neg_min]

        elif self.training_type == "semi_hard_batch_learning":

            for anchor_iter in range(len(anchor_mfs)):

                anchor_label = anchor_labels[anchor_iter]

                if anchor_label == 1:
                    pos_pool = [id for id in indices_1 if id != anchor_iter]
                    neg_pool = indices_0
                else:
                    pos_pool = [id for id in indices_0 if id != anchor_iter]
                    neg_pool = indices_1

                # pos -> random
                # neg -> semi-hard
                pos_idx = pos_pool[torch.randint(len(pos_pool), (1,)).item()]

                distances_pos = distances[anchor_iter, pos_idx]
                distances_neg = distances[anchor_iter, neg_pool]

                semi_hard_mask = (distances_neg > distances_pos) & (distances_neg < distances_pos + self.margin)
                semi_hard_idx = semi_hard_mask.nonzero(as_tuple=True)[0].tolist()

                if semi_hard_mask.any():
                    # pick first semi-hard (or random among them)
                    valid_ids = [neg_pool[i] for i in semi_hard_idx]
                    neg_idx = valid_ids[torch.randint(len(valid_ids), (1,)).item()]
                else:
                    # fallback to closest negative
                    neg_idx = neg_pool[distances_neg.argmin().item()]

                positive_mfs[anchor_iter] = anchor_mfs[pos_idx]
                negative_mfs[anchor_iter] = anchor_mfs[neg_idx]

                positive_mfs_distances[anchor_iter] = distances_pos
                negative_mfs_distances[anchor_iter] = distances[anchor_iter, neg_idx]

        else:
            raise Exception("Training type not implemented")

        return anchor_mfs, positive_mfs, positive_mfs_distances, negative_mfs, negative_mfs_distances, anchor_labels