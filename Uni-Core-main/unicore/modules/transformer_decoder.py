import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
from typing import Optional

import paddle
from paddle_utils import *

from . import LayerNorm, TransformerDecoderLayer
from .transformer_encoder import relative_position_bucket


def fill_with_neg_inf(t):
    return t.fill_(value=float("-inf"))


def bulid_future_mask(seq_len):
    return paddle.triu(
        x=fill_with_neg_inf(paddle.zeros(shape=[seq_len, seq_len])), diagonal=1
    )


class TransformerDecoder(paddle.nn.Layer):
    def __init__(
        self,
        decoder_layers: int = 6,
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
        auto_regressive: bool = True,
    ) -> None:
        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        self.auto_regressive = auto_regressive
        if self.auto_regressive:
            self._future_mask = bulid_future_mask(self.max_seq_len)
        else:
            self._future_mask = None
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None
        self.layers = paddle.nn.LayerList(
            sublayers=[
                TransformerDecoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(decoder_layers)
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

    def get_future_mask(self, x, attn_mask):
        if not self.auto_regressive:
            return attn_mask
        if self._future_mask.place != x.place:
            self._future_mask = self._future_mask.to(x.place)
        if self._future_mask.dtype != x.dtype:
            self._future_mask = self._future_mask.astype(dtype=x.dtype)
        if attn_mask is None:
            ret = self._future_mask[: x.shape[1], : x.shape[1]]
            ret = (
                ret.contiguous()
                .unsqueeze(axis=0)
                .tile(repeat_times=[x.shape[0] * self.attention_heads, 1, 1])
            )
            return ret
        else:
            assert list(tuple(attn_mask.shape)) == [
                x.shape[0] * self.attention_heads,
                x.shape[1],
                x.shape[1],
            ]
            return attn_mask + self._future_mask[: x.shape[1], : x.shape[1]]

    def forward(
        self,
        emb,
        encoder_out: Optional[paddle.Tensor] = None,
        padding_mask: Optional[paddle.Tensor] = None,
        encoder_padding_mask: Optional[paddle.Tensor] = None,
        attn_mask: Optional[paddle.Tensor] = None,
        encoder_attn_mask: Optional[paddle.Tensor] = None,
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
        if self.auto_regressive:
            attn_mask = self.get_future_mask(x, attn_mask)
        if attn_mask is not None and padding_mask is not None:
            attn_mask = attn_mask.view(x.shape[0], -1, seq_len, seq_len)
            attn_mask.masked_fill_(
                mask=padding_mask.unsqueeze(axis=1).unsqueeze(axis=2).to("bool"),
                value=float("-inf"),
            )
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None
        for layer in self.layers:
            x = layer(
                x,
                encoder_out=encoder_out,
                padding_mask=padding_mask,
                attn_bias=attn_mask,
                encoder_padding_mask=encoder_padding_mask,
                encoder_attn_bias=encoder_attn_mask,
            )
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        return x
