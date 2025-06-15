import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
from functools import lru_cache

import numpy as np
import paddle
from paddle_utils import *
from scipy.spatial import distance_matrix
from unicore.data import BaseWrapperDataset


class DistanceDataset(BaseWrapperDataset):
    def __init__(self, dataset, p=2):
        super().__init__(dataset)
        self.dataset = dataset
        self.p = p

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pos = self.dataset[idx].view(-1, 3).numpy()
        dist = distance_matrix(pos, pos, self.p).astype(np.float32)
        return paddle.to_tensor(data=dist)


class EdgeTypeDataset(BaseWrapperDataset):
    def __init__(self, dataset: paddle.io.Dataset, num_types: int):
        self.dataset = dataset
        self.num_types = num_types

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        node_input = self.dataset[index].clone()
        offset = node_input.view(-1, 1) * self.num_types + node_input.view(1, -1)
        return offset
