import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
import math
from typing import Optional

import paddle
from paddle_utils import *

from . import LayerNorm, TransformerEncoderLayer


def init_bert_params(module):
    if not getattr(module, "can_global_init", True):
        return

    def normal_(data):
        return paddle.normal(mean=0.0, std=0.02, shape=data.shape).astype(data.dtype).to(data.place)

    if isinstance(module, paddle.nn.Linear):
        module.weight.set_value(normal_(module.weight))
        if module.bias is not None:
            module.bias.set_value(paddle.zeros_like(module.bias))
    if isinstance(module, paddle.nn.Embedding):
        module.weight.set_value(normal_(module.weight))
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    sign = paddle.sign(x=relative_position)
    num_buckets //= 2
    n = paddle.abs(x=relative_position)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    max_bucket_val = num_buckets - 1 - max_exact
    val_if_large = max_exact + paddle.ceil(
        x=paddle.log(x=n.astype(dtype="float32") / max_exact)
        / math.log((max_distance - 1) / max_exact)
        * max_bucket_val
    ).astype(dtype="int64")
    val_if_large = paddle_min(
        val_if_large, paddle.full_like(x=val_if_large, fill_value=num_buckets - 1)
    )
    ret = paddle.where(condition=is_small, x=n, y=val_if_large) * sign
    return ret


class TransformerEncoder(paddle.nn.Layer):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        rel_pos: bool = True,
        rel_pos_bins: int = 32,
        max_rel_pos: int = 128,
        post_ln: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None
        self.layers = paddle.nn.LayerList(
            sublayers=[
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.rel_pos = rel_pos
        if self.rel_pos:
            assert rel_pos_bins % 2 == 0
            self.rel_pos_bins = rel_pos_bins
            self.max_rel_pos = max_rel_pos
            self.relative_attention_bias = Embedding(
                num_embeddings=self.rel_pos_bins, embedding_dim=self.attention_heads
            )
            seq_len = self.max_seq_len
            context_position = paddle.arange(dtype="int64", end=seq_len)[:, None]
            memory_position = paddle.arange(dtype="int64", end=seq_len)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = relative_position_bucket(
                relative_position,
                num_buckets=self.rel_pos_bins,
                max_distance=self.max_rel_pos,
            )
            self.rp_bucket -= self.rp_bucket._min()

    def get_rel_pos_bias(self, x):
        if self.rp_bucket.place != x.place:
            self.rp_bucket = self.rp_bucket.to(x.place)
        seq_len = x.shape[1]
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = paddle.nn.functional.embedding(
            x=rp_bucket, weight=self.relative_attention_bias.weight
        )
        values = values.transpose(perm=[2, 0, 1])
        return values.contiguous()

    def forward(
        self,
        emb: paddle.Tensor,
        attn_mask: Optional[paddle.Tensor] = None,
        padding_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        seq_len = emb.shape[1]
        x = self.emb_layer_norm(emb)
        x = paddle.nn.functional.dropout(
            x=x, p=self.emb_dropout, training=self.training
        )
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(axis=-1).astype(dtype=x.dtype))
        rel_pos_bias = (
            self.get_rel_pos_bias(x).tile(repeat_times=[x.shape[0], 1, 1])
            if self.rel_pos
            else None
        )
        if attn_mask is None:
            attn_mask = rel_pos_bias
        elif rel_pos_bias is not None:
            attn_mask += rel_pos_bias
        if attn_mask is not None and padding_mask is not None:
            attn_mask = attn_mask.view(x.shape[0], -1, seq_len, seq_len)
            attn_mask.masked_fill_(
                mask=padding_mask.unsqueeze(axis=1).unsqueeze(axis=2).to("bool"),
                value=float("-inf"),
            )
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask, attn_bias=attn_mask)
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        return x
