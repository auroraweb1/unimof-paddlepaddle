from functools import lru_cache

import numpy as np
from unicore.data import BaseWrapperDataset


class LatticeNormalizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, abc, angles):
        super().__init__(dataset)
        self.dataset = dataset
        self.abc = abc
        self.angles = angles

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        abc = np.array(self.dataset[idx][self.abc])
        angles = np.array(self.dataset[idx][self.angles])
        lattices = normalize_v2(abc, angles)
        return lattices


def normalize(abc, angles):
    indices = np.argsort(abc)
    abc = abc[indices]
    angles = angles[indices]
    angles = [min(item, 180.0 - item) for item in angles]
    angles = np.array(angles) / 180.0 * np.pi
    lattices = np.concatenate([abc, angles]).astype(np.float32)
    return lattices


def normalize_v2(abc, angles):
    angles = np.array(angles) / 180.0 * np.pi
    lattices = angles.astype(np.float32)
    return lattices
