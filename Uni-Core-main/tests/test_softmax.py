import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
import paddle
from paddle_utils import *
from unicore.modules import softmax_dropout


def gen_attn_mask(mask, neg_inf):
    assert neg_inf < -10000.0
    attn_mask = paddle.zeros_like(x=mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask


def normal_softmax(a, mask, bias):
    return paddle.nn.functional.softmax(x=a + mask + bias, axis=-1)


def fused_softmax(a, mask, bias):
    return softmax_dropout(a, 0, True, mask=mask, bias=bias)


def wrap_forward_backward(func, a1, mask, bias1):
    a = a1.clone()
    bias = bias1.clone()
    a.stop_gradient = not True
    bias.stop_gradient = not True
    output = func(a, mask, bias)
    o = output.astype(dtype="float32").sum()
    o.backward()
    return output, a.grad, bias.grad


def check_diff(a, b, name, eps=0.001):
    assert (a - b).abs()._max() < eps, "name {}, diff {}".format(
        name, (a - b).abs()._max()
    )


def test_softmax():
    n_batch = 4
    n_heads = 8
    n_query = 128
    test_dims = [64, 128, 256, 512, 1024, 1536, 2048]
    test_dtype = ["float32", "float16", "bfloat16"]
    test_device = device2str("cuda")
    for last_dim in test_dims:
        for dtype in test_dtype:
            x = paddle.rand(shape=[n_batch, n_heads, n_query, last_dim], dtype=dtype)
            mask = gen_attn_mask(
                (
                    paddle.rand(shape=[n_batch, 1, 1, last_dim], dtype=dtype) > 0.2
                ).astype(x.dtype),
                -30000.0,
            )
            bias = paddle.rand(shape=[n_batch, n_heads, n_query, last_dim], dtype=dtype)
            out_a1, out_b1, out_c1 = wrap_forward_backward(
                normal_softmax, x, mask, bias
            )
            out_a2, out_b2, out_c2 = wrap_forward_backward(fused_softmax, x, mask, bias)
            check_diff(out_a1, out_a2, "output")
            check_diff(out_b1, out_b2, "grad_input")
            check_diff(out_c1, out_c2, "grad_bias")


def test_tri_softmax1():
    n_batch = 2
    n_groups = 32
    n_heads = 8
    n_query = 128
    test_dims = [64, 128, 256, 512, 1024, 1536, 2048]
    test_dtype = ["float32", "float16", "bfloat16"]
    test_device = device2str("cuda")
    for last_dim in test_dims:
        for dtype in test_dtype:
            x = paddle.rand(
                shape=[n_batch, n_groups, n_heads, n_query, last_dim], dtype=dtype
            )
            mask = gen_attn_mask(
                (
                    paddle.rand(shape=[n_batch, n_groups, 1, 1, last_dim], dtype=dtype)
                    > 0.2
                ).astype(x.dtype),
                -30000.0,
            )
            bias = paddle.rand(shape=[1, 1, n_heads, n_query, last_dim], dtype=dtype)
            out_a1, out_b1, out_c1 = wrap_forward_backward(
                normal_softmax, x, mask, bias
            )
            out_a2, out_b2, out_c2 = wrap_forward_backward(fused_softmax, x, mask, bias)
            check_diff(out_a1, out_a2, "output")
            check_diff(out_b1, out_b2, "grad_input")
            check_diff(out_c1, out_c2, "grad_bias")


def test_tri_softmax2():
    n_batch = 2
    n_groups = 32
    n_heads = 8
    n_query = 128
    test_dims = [64, 128, 256, 512, 1024, 1536, 2048]
    test_dtype = ["float32", "float16", "bfloat16"]
    test_device = device2str("cuda")
    for last_dim in test_dims:
        for dtype in test_dtype:
            x = paddle.rand(
                shape=[n_batch, n_groups, n_heads, n_query, last_dim], dtype=dtype
            )
            mask = gen_attn_mask(
                (
                    paddle.rand(
                        shape=[n_batch, n_groups, n_heads, 1, last_dim], dtype=dtype
                    )
                    > 0.2
                ).astype(x.dtype),
                -30000.0,
            )
            bias = paddle.rand(
                shape=[1, n_groups, n_heads, n_query, last_dim], dtype=dtype
            )
            out_a1, out_b1, out_c1 = wrap_forward_backward(
                normal_softmax, x, mask, bias
            )
            out_a2, out_b2, out_c2 = wrap_forward_backward(fused_softmax, x, mask, bias)
            check_diff(out_a1, out_a2, "output")
            check_diff(out_b1, out_b2, "grad_input")
            check_diff(out_c1, out_c2, "grad_bias")
