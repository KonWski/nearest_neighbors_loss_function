import torch

class FocalBatchShaper():

    def __init__(self, device):
        self.device = device    

    def shape_batch(self, anchor_mfs, anchor_labels):

        indices_1 = (anchor_labels == 1).nonzero()[:, 0].tolist()
        indices_0 = (anchor_labels == 0).nonzero()[:, 0].tolist()

        n_indices_1 = len(indices_1)
        n_indices_0 = len(indices_0)

        anchors_l = []
        anchors_r = []
        labels = []

        # majority positives (positive pairs)
        if n_indices_1 >= 2:
            for i in range(n_indices_1):
                for j in range(i + 1, n_indices_1):
                    anchors_l.append(anchor_mfs[indices_1[i]])
                    anchors_r.append(anchor_mfs[indices_1[j]])
                    labels.append(1)

        # majority positives (negative pairs)
        if n_indices_0 >= 2:
            for i in range(n_indices_0):
                for j in range(i + 1, len(indices_0)):
                    anchors_l.append(anchor_mfs[indices_0[i]])
                    anchors_r.append(anchor_mfs[indices_0[j]])
                    labels.append(1)

        # majority positives (negative pairs)
        if n_indices_0 >= 2:
            for i in range(n_indices_0):
                anchors_l.append(anchor_mfs[indices_0[i]])
                anchors_r.append(anchor_mfs[indices_0[n_indices_0 - i]])
                labels.append(1)


        # negative pairs
        for i in indices_1:
            for j in indices_0:
                anchors_l.append(anchor_mfs[i])
                anchors_r.append(anchor_mfs[j])
                labels.append(0)

        anchors_l = torch.stack(anchors_l).to(self.device)
        anchors_r = torch.stack(anchors_r).to(self.device)
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)

        return anchors_l, anchors_r, labels