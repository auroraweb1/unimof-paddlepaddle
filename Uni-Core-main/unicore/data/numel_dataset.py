import numpy as np
import paddle

from . import BaseWrapperDataset


class NumelDataset(BaseWrapperDataset):
    def __init__(self, dataset, reduce=False):
        super().__init__(dataset)
        self.reduce = reduce

    def __getitem__(self, index):
        item = self.dataset[index]
        if paddle.is_tensor(x=item):
            return item.size
        else:
            return np.size(item)

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.reduce:
            return sum(samples)
        else:
            return paddle.to_tensor(data=samples)
