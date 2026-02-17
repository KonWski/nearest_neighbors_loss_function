import torch

class BatchShaper():

    def __init__(self, training_type = "hard_batch_learning"):
        self.training_type = training_type


    def shape_batch(self, anchor_mfs, anchor_labels):

        indices_1 = (anchor_labels == 1).nonzero()[:,0].tolist()
        indices_0 = (anchor_labels == 0).nonzero()[:,0].tolist()
        
        anchor_mfs_1 = anchor_mfs[indices_1,:]
        anchor_mfs_0 = anchor_mfs[indices_0,:]
        distances = torch.cdist(anchor_mfs, anchor_mfs)

        positive_mfs = []
        negative_mfs = []

        if self.training_type == "hard_batch_learning":

            for anchor_iter in range(len(anchor_mfs)):

                anchor_label = anchor_labels[anchor_iter]

                if anchor_label == 0:
                    distances_pos = distances[anchor_iter, indices_0]
                    distances_neg = distances[anchor_iter, indices_1]

                    id_distance_pos_max = distances_pos.argmax().item()
                    id_distance_neg_min = distances_neg.argmin().item()
                    positive_mfs.append(anchor_mfs_0[id_distance_pos_max, :])
                    negative_mfs.append(anchor_mfs_1[id_distance_neg_min, :])

                elif anchor_label == 1:
                    distances_pos = distances[anchor_iter, indices_1]
                    distances_neg = distances[anchor_iter, indices_0]
                    id_distance_pos_max = distances_pos.argmax().item()
                    id_distance_neg_min = distances_neg.argmin().item()
                    positive_mfs.append(anchor_mfs_1[id_distance_pos_max, :])
                    negative_mfs.append(anchor_mfs_0[id_distance_neg_min, :])

        elif self.training_type == "semi_hard_batch_learning":

            for anchor_iter in range(len(anchor_mfs)):

                anchor_label = anchor_labels[anchor_iter]

                if anchor_label == 1:
                    pos_pool = indices_1[indices_1 != anchor_iter]
                    neg_pool = indices_0
                else:
                    pos_pool = indices_0[indices_0 != anchor_iter]
                    neg_pool = indices_1

                # pos -> random
                # neg -> semi-hard
                pos_idx = pos_pool[torch.randint(len(pos_pool), (1,)).item()]

                distances_pos = distances[anchor_iter, pos_idx]
                distances_neg = distances[anchor_iter, neg_pool]

                semi_hard_mask = (distances_neg > distances_pos) & (distances_neg < distances_pos + self.margin)

                if semi_hard_mask.any():
                    # pick first semi-hard (or random among them)
                    valid_ids = neg_pool[semi_hard_mask.nonzero(as_tuple=True)[0]]
                    neg_idx = valid_ids[torch.randint(len(valid_ids), (1,)).item()]
                else:
                    # fallback to closest negative
                    neg_idx = neg_pool[distances_neg.argmin().item()]

                positive_mfs.append(anchor_mfs[pos_idx])
                negative_mfs.append(anchor_mfs[neg_idx])

        else:
            raise Exception("Training type not implemented")

        positive_mfs = torch.stack(positive_mfs, dim=0)
        negative_mfs = torch.stack(negative_mfs, dim=0)

        return anchor_mfs, positive_mfs, negative_mfs, anchor_labels