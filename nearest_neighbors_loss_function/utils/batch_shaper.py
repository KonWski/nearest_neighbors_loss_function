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

        if self.training_type == "hard_batch_learning":

            positive_mfs = []
            negative_mfs = []

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

        else:
            raise Exception("Training type not implemented")

        positive_mfs = torch.stack(positive_mfs, dim=0)
        negative_mfs = torch.stack(negative_mfs, dim=0)

        return anchor_mfs, positive_mfs, negative_mfs, anchor_labels
