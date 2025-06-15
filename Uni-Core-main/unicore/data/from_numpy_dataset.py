from functools import lru_cache

import paddle

from . import BaseWrapperDataset


class FromNumpyDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return paddle.to_tensor(data=self.dataset[idx])
