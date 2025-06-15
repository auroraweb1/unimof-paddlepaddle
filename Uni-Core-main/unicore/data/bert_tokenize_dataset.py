from functools import lru_cache

import numpy as np
import paddle
from tokenizers import BertWordPieceTokenizer

from . import BaseWrapperDataset, LRUCacheDataset


class BertTokenizeDataset(BaseWrapperDataset):
    def __init__(
        self, dataset: paddle.io.Dataset, dict_path: str, max_seq_len: int = 512
    ):
        self.dataset = dataset
        self.tokenizer = BertWordPieceTokenizer(dict_path, lowercase=True)
        self.max_seq_len = max_seq_len

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def __getitem__(self, index: int):
        raw_str = self.dataset[index]
        raw_str = raw_str.replace("<unk>", "[UNK]")
        output = self.tokenizer.encode(raw_str)
        ret = paddle.to_tensor(data=output.ids, dtype="float32").astype(dtype="int64")
        if ret.shape[0] > self.max_seq_len:
            ret = ret[: self.max_seq_len]
        return ret
