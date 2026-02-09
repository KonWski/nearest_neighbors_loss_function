from torch.utils.data import BatchSampler
import random

class BalancedSampler(BatchSampler):
    def __init__(self, labels, minority_class, minority_per_batch, 
                 batch_size, shuffle=True, seed = 123):
        self.labels = labels
        self.minority_class = minority_class
        self.minority_per_batch = minority_per_batch
        self.batch_size = batch_size
        self.majority_per_batch = batch_size - minority_per_batch
        self.shuffle = shuffle
        self.seed = seed

        self.minority_indices = [
            i for i, y in enumerate(labels) if y == minority_class
        ]
        self.majority_indices = [
            i for i, y in enumerate(labels) if y != minority_class
        ]


    def __iter__(self):

        if self.shuffle:
            random.shuffle(self.minority_indices)
            random.shuffle(self.majority_indices)

        min_ptr, maj_ptr = 0, 0

        while (
            min_ptr + self.minority_per_batch <= len(self.minority_indices)
            and maj_ptr + self.majority_per_batch <= len(self.majority_indices)
        ):
            batch = (
                self.minority_indices[
                    min_ptr : min_ptr + self.minority_per_batch
                ]
                + self.majority_indices[
                    maj_ptr : maj_ptr + self.majority_per_batch
                ]
            )

            yield batch

            min_ptr += self.minority_per_batch
            maj_ptr += self.majority_per_batch

    def __len__(self):
        return min(
            len(self.minority_indices) // self.minority_per_batch,
            len(self.majority_indices) // self.majority_per_batch,
        )