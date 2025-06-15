from functools import lru_cache

import numpy as np
import paddle
from unicore.data import Dictionary, data_utils

from . import BaseWrapperDataset, LRUCacheDataset


class MaskTokensDataset(BaseWrapperDataset):
    @classmethod
    def apply_mask(cls, dataset: paddle.io.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return LRUCacheDataset(
            cls(dataset, *args, **kwargs, return_masked_tokens=False)
        ), LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True))

    def __init__(
        self,
        dataset: paddle.io.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0
        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        if random_token_prob > 0.0:
            weights = np.ones(len(self.vocab))
            weights[vocab.special_index()] = 0
            self.weights = weights / weights.sum()
        self.epoch = None

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, epoch, index):
            item = self.dataset[index]
            sz = len(item)
            assert sz > 2
            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx
            )
            mask = np.full(sz, False)
            num_mask = int(self.mask_prob * (sz - 2) + np.random.rand())
            mask_idc = np.random.choice(sz - 2, num_mask, replace=False) + 1
            mask[mask_idc] = True
            if self.return_masked_tokens:
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[paddle.to_tensor(data=mask.astype(np.uint8)) == 1]
                return paddle.to_tensor(data=new_item)
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
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab), num_rand, p=self.weights
                    )
            return paddle.to_tensor(data=new_item)
