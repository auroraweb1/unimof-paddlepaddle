from functools import lru_cache

import paddle

from . import UnicoreDataset


class RawLabelDataset(UnicoreDataset):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return paddle.to_tensor(data=samples)


class RawArrayDataset(UnicoreDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, "collater"):
            return self.dataset.collater(samples)
        else:
            return paddle.io.dataloader.collate.default_collate_fn(batch=samples)


class RawNumpyDataset(UnicoreDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return paddle.to_tensor(data=self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, "collater"):
            return self.dataset.collater(samples)
        else:
            return paddle.io.dataloader.collate.default_collate_fn(batch=samples)
