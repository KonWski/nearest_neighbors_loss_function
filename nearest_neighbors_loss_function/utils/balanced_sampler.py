from torch.utils.data import BatchSampler
import random

class BalancedSampler(BatchSampler):
    def __init__(self, labels, n_minority_samples, minority_class, minority_per_batch, 
                 batch_size, n_batches, shuffle=True, seed = 123):
        self.labels = labels
        self.n_minority_samples = n_minority_samples
        self.minority_class = minority_class
        self.minority_per_batch = minority_per_batch
        self.n_extra_minority_batches = self.n_minority_samples - (self.n_batches * minority_per_batch)
        self.majority_per_batch = batch_size - minority_per_batch
        self.batch_size = batch_size
        self.n_batches = n_batches
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
            random.seed(self.seed)
            random.shuffle(self.minority_indices)
            random.shuffle(self.majority_indices)
            self.extra_minority_batch_ids = set(random.sample([i for i in range(self.n_batches)], self.n_extra_minority_batches))

        min_ptr, maj_ptr = 0, 0
        batch_id = 0

        while (
            min_ptr + self.minority_per_batch <= len(self.minority_indices)
            and maj_ptr + self.majority_per_batch <= len(self.majority_indices)
        ):
            extra_minority_iter = 0
            if batch_id in self.extra_minority_batch_ids:
                extra_minority_iter = 1

            batch = (
                self.minority_indices[
                    min_ptr : min_ptr + self.minority_per_batch + extra_minority_iter
                ]
                + self.majority_indices[
                    maj_ptr : maj_ptr + self.majority_per_batch
                ]
            )
            
            batch_id += 1
            yield batch

            min_ptr += self.minority_per_batch + extra_minority_iter
            maj_ptr += self.majority_per_batch


    def __len__(self):
        return self.n_batches