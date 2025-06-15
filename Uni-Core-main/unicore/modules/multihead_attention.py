import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
from typing import Dict, Optional

import paddle
from paddle_utils import *

from .softmax_dropout import softmax_dropout


class SelfMultiheadAttention(paddle.nn.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, scaling_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5
        self.in_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=embed_dim * 3, bias_attr=bias
        )
        self.out_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias_attr=bias
        )

    def forward(
        self,
        query,
        key_padding_mask: Optional[paddle.Tensor] = None,
        attn_bias: Optional[paddle.Tensor] = None,
        return_attn: bool = False,
    ) -> paddle.Tensor:
        bsz, tgt_len, embed_dim = tuple(query.shape)
        assert embed_dim == self.embed_dim
        q, k, v = self.in_proj(query).chunk(chunks=3, axis=-1)
        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(
                perm=dim2perm(
                    q.view(bsz, tgt_len, self.num_heads, self.head_dim).ndim, 1, 2
                )
            )
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(
                    perm=dim2perm(
                        k.view(bsz, -1, self.num_heads, self.head_dim).ndim, 1, 2
                    )
                )
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(
                    perm=dim2perm(
                        v.view(bsz, -1, self.num_heads, self.head_dim).ndim, 1, 2
                    )
                )
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        assert k is not None
        src_len = k.shape[1]
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len
        attn_weights = paddle.bmm(x=q, y=k.transpose(perm=dim2perm(k.ndim, 1, 2)))
        assert list(tuple(attn_weights.shape)) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                mask=key_padding_mask.unsqueeze(axis=1).unsqueeze(axis=2).to("bool"),
                value=float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if not return_attn:
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, bias=attn_bias
            )
        else:
            attn_weights += attn_bias
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, inplace=False
            )
        o = paddle.bmm(x=attn, y=v)
        assert list(tuple(o.shape)) == [bsz * self.num_heads, tgt_len, self.head_dim]
        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(
                perm=dim2perm(
                    o.view(bsz, self.num_heads, tgt_len, self.head_dim).ndim, 1, 2
                )
            )
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn


class CrossMultiheadAttention(paddle.nn.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, scaling_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5
        self.q_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias_attr=bias
        )
        self.k_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias_attr=bias
        )
        self.v_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias_attr=bias
        )
        self.out_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias_attr=bias
        )

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask: Optional[paddle.Tensor] = None,
        attn_bias: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        bsz, tgt_len, embed_dim = tuple(query.shape)
        assert embed_dim == self.embed_dim
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(
                perm=dim2perm(
                    q.view(bsz, tgt_len, self.num_heads, self.head_dim).ndim, 1, 2
                )
            )
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(
                    perm=dim2perm(
                        k.view(bsz, -1, self.num_heads, self.head_dim).ndim, 1, 2
                    )
                )
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(
                    perm=dim2perm(
                        v.view(bsz, -1, self.num_heads, self.head_dim).ndim, 1, 2
                    )
                )
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        assert k is not None
        src_len = k.shape[1]
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len
        attn_weights = paddle.bmm(x=q, y=k.transpose(perm=dim2perm(k.ndim, 1, 2)))
        assert list(tuple(attn_weights.shape)) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                mask=key_padding_mask.unsqueeze(axis=1).unsqueeze(axis=2).to("bool"),
                value=float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn = softmax_dropout(
            attn_weights, self.dropout, self.training, bias=attn_bias
        )
        o = paddle.bmm(x=attn, y=v)
        assert list(tuple(o.shape)) == [bsz * self.num_heads, tgt_len, self.head_dim]
        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(
                perm=dim2perm(
                    o.view(bsz, self.num_heads, tgt_len, self.head_dim).ndim, 1, 2
                )
            )
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        return o
