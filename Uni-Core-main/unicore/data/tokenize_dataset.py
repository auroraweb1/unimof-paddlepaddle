from functools import lru_cache

import paddle
from unicore.data import Dictionary

from . import BaseWrapperDataset


class TokenizeDataset(BaseWrapperDataset):
    def __init__(
        self, dataset: paddle.io.Dataset, dictionary: Dictionary, max_seq_len: int = 512
    ):
        self.dataset = dataset
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        raw_data = self.dataset[index]
        assert len(raw_data) < self.max_seq_len and len(raw_data) > 0
        return paddle.to_tensor(data=self.dictionary.vec_index(raw_data)).astype(
            dtype="int64"
        )
