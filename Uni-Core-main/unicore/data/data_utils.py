import contextlib
import logging

import numpy as np
import paddle

logger = logging.getLogger(__name__)


def collate_tokens(
    values, pad_idx, left_pad=False, pad_to_length=None, pad_to_multiple=1
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.shape[0] for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = paddle.full(
        shape=[len(values), size],
        fill_value=pad_idx,
        dtype=values[0].dtype
    )

    def copy_tensor(src, dst):
        assert dst.size == src.size
        paddle.assign(src, output=dst)

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
    res = paddle.full(
        shape=[len(values), size, size],
        fill_value=pad_idx,
        dtype=values[0].dtype
    )

    def copy_tensor(src, dst):
        assert dst.size == src.size
        paddle.assign(src, output=dst)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size - len(v) :, size - len(v) :]
            if left_pad
            else res[i][: len(v), : len(v)],
        )
    return res


def collate_dict(values, dim=0):
    if len(values) <= 0:
        return values
    ret = {}
    keys = values[0].keys()
    for key in keys:
        ret[key] = paddle.stack(x=[v[key] for v in values], axis=dim)
    return ret


def str_hash(text: str):
    hash = 0
    for ch in text:
        hash = (hash * 281 ^ ord(ch) * 997) & 4294967295
    return hash


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds, key=None):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return

    def check_seed(s):
        assert type(s) == int or type(s) == np.int32 or type(s) == np.int64

    check_seed(seed)
    if len(addl_seeds) > 0:
        for s in addl_seeds:
            check_seed(s)
        seed = int(hash((seed, *addl_seeds)) % 100000000.0)
    if key is not None:
        seed = int(hash((seed, str_hash(key))) % 100000000.0)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def batch_by_size(indices, batch_size=None, required_batch_size_multiple=1):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        batch_size (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
    """
    batch_size = batch_size if batch_size is not None else 1
    bsz_mult = required_batch_size_multiple
    step = (batch_size + bsz_mult - 1) // bsz_mult * bsz_mult
    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)
    num_batches = (len(indices) + step - 1) // step
    steps = np.arange(num_batches - 1) + 1
    steps *= step
    batch_indices = np.split(indices, steps)
    assert len(batch_indices) == num_batches
    assert tuple(batch_indices[0].shape)[0] <= step
    return batch_indices
