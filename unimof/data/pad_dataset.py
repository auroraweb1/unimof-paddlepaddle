from functools import lru_cache

import paddle
from unicore.data import BaseWrapperDataset


def copy_tensor(src, dst):
    assert dst.size == src.size
    paddle.assign(src, output=dst)


def collate_tokens(
    values, pad_idx, left_pad=False, pad_to_length=None, pad_to_multiple=1
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.shape[0] for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = paddle.full(shape=[len(values), size], fill_value=pad_idx, dtype=values[0].dtype)
    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def collate_tokens_2d(
    values, pad_idx, left_pad=False, pad_to_length=None, pad_to_multiple=1
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.shape[0] for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = paddle.full(shape=[len(values), size, size], fill_value=pad_idx, dtype=values[0].dtype)
    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size - len(v) :, size - len(v) :]
            if left_pad
            else res[i][: len(v), : len(v)],
        )
    return res


def collate_tokens_coords(
    values, pad_idx, left_pad=False, pad_to_length=None, pad_to_multiple=1
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.shape[0] for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = paddle.full(shape=[len(values), size, 3], fill_value=pad_idx, dtype=values[0].dtype)
    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res


class PadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_tokens(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8
        )


class LeftPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=True)


class RightPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=False)


class RightPadDataset2D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_tokens_2d(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8
        )


class RightPadDatasetCoord(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_tokens_coords(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8
        )


class PrependAndAppend2DDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            h, w = item.shape[-2], item.shape[-1]
            new_item = paddle.full(shape=(h + 2, w + 2), fill_value=self.token).astype(
                dtype=item.dtype
            )
            new_item[1:-1, 1:-1] = item
            return new_item
        return item
