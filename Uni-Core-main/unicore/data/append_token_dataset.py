from functools import lru_cache

import numpy as np
import paddle

from . import BaseWrapperDataset


class AppendTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            item = paddle.concat(
                x=[
                    item,
                    paddle.full_like(x=item[0], fill_value=self.token).unsqueeze(
                        axis=0
                    ),
                ],
                axis=0,
            )
        return item
