import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
from functools import lru_cache

import numpy as np
import paddle
from paddle_utils import *
from unicore.data import BaseWrapperDataset, Dictionary, data_utils


class MaskPointsChunkDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: paddle.io.Dataset,
        coord_dataset: paddle.io.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        noise_type: str,
        noise: float = 1.0,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        chunk_mask_prob: float = 0.5,
        chunk_ratio: float = 0.5,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0
        assert 0.0 <= chunk_mask_prob <= 1.0
        self.dataset = dataset
        self.coord_dataset = coord_dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.noise_type = noise_type
        self.noise = noise
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.chunk_mask_prob = chunk_mask_prob
        self.chunk_ratio = chunk_ratio
        if random_token_prob > 0.0:
            weights = np.ones(len(self.vocab))
            weights[vocab.special_index()] = 0
            self.weights = weights / weights.sum()
        self.epoch = None
        if self.noise_type == "trunc_normal":
            self.noise_f = lambda num_mask: np.clip(
                np.random.randn(num_mask, 3) * self.noise,
                a_min=-self.noise * 2.0,
                a_max=self.noise * 2.0,
            )
        elif self.noise_type == "normal":
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise
        elif self.noise_type == "uniform":
            self.noise_f = lambda num_mask: np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 3)
            )
        else:
            self.noise_f = lambda num_mask: 0.0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.coord_dataset.set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        ret = {}
        with data_utils.numpy_seed(self.seed, epoch, index):
            item = self.dataset[index]
            coord = self.coord_dataset[index]
            sz = len(item)
            assert sz > 0
            if np.random.rand() > self.rand_mask_prob:
                num_mask = int(self.mask_prob * sz + np.random.rand())
                mask_idc = np.random.choice(sz, num_mask, replace=False)
                mask = np.full(sz, False)
                mask[mask_idc] = True
                ret["targets"] = np.full(len(mask), self.pad_idx)
                ret["targets"][mask] = item[mask]
                ret["targets"] = paddle.to_tensor(data=ret["targets"]).astype(
                    dtype="int64"
                )
                rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
                if rand_or_unmask_prob > 0.0:
                    rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                    if self.random_token_prob == 0.0:
                        unmask = rand_or_unmask
                        rand_mask = None
                    elif self.leave_unmasked_prob == 0.0:
                        unmask = None
                        rand_mask = rand_or_unmask
                    else:
                        unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                        decision = np.random.rand(sz) < unmask_prob
                        unmask = rand_or_unmask & decision
                        rand_mask = rand_or_unmask & ~decision
                else:
                    unmask = rand_mask = None
                if unmask is not None:
                    mask = mask ^ unmask
                new_item = np.copy(item)
                new_item[mask] = self.mask_idx
                num_mask = mask.astype(np.int32).sum()
                new_coord = np.copy(coord)
                new_coord[mask, :] += self.noise_f(num_mask)
                if rand_mask is not None:
                    num_rand = rand_mask.sum()
                    if num_rand > 0:
                        new_item[rand_mask] = np.random.choice(
                            len(self.vocab), num_rand, p=self.weights
                        )
                ret["atoms"] = paddle.to_tensor(data=new_item).astype(dtype="int64")
                ret["coordinates"] = paddle.to_tensor(data=new_coord).astype(
                    dtype="float32"
                )
            else:
                minx, miny, minz = coord._min(axis=0)
                maxx, maxy, maxz = coord._max(axis=0)
            return ret
