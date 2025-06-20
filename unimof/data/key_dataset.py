from functools import lru_cache

import numpy as np
import paddle
from unicore.data import BaseWrapperDataset


class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx][self.key]


class ToTorchDataset(BaseWrapperDataset):
    def __init__(self, dataset, dtype="float32"):
        super().__init__(dataset)
        self.dtype = dtype

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        d = np.array(self.dataset[idx], dtype=self.dtype)
        return paddle.to_tensor(data=d)


class NumericalTransformDataset(BaseWrapperDataset):
    def __init__(self, dataset, ops="log1p"):
        super().__init__(dataset)
        self.dataset = dataset
        self.ops = ops

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.ops == "log1p":
            d = np.array(self.dataset[idx], dtype="float64")
            d = np.log1p(d).astype("float32")
        else:
            d = np.array(self.dataset[idx])
        return d


class FlattenDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        dd = self.dataset[index].copy()
        dd = np.array(dd).reshape(-1)
        return dd
