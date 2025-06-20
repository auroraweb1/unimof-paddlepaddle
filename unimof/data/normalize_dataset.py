from functools import lru_cache

import numpy as np
from unicore.data import BaseWrapperDataset


class NormalizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, coordinates):
        self.dataset = dataset
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        coordinates = dd[self.coordinates]
        coordinates = coordinates - coordinates.mean(axis=0)
        dd[self.coordinates] = coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
