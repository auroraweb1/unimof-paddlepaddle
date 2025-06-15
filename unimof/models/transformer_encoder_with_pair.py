import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
import math
from typing import Optional

import paddle
from paddle_utils import *
from unicore.modules import LayerNorm, TransformerEncoderLayer


class TransformerEncoderWithPair(paddle.nn.Layer):
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
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
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
        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None
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

    def forward(
        self,
        emb: paddle.Tensor,
        attn_mask: Optional[paddle.Tensor] = None,
        padding_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        bsz = emb.shape[0]
        seq_len = emb.shape[1]
        x = self.emb_layer_norm(emb)
        x = paddle.nn.functional.dropout(
            x=x, p=self.emb_dropout, training=self.training
        )
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(axis=-1).astype(dtype=x.dtype))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask
        mask_pos = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                attn_mask = attn_mask.view(x.shape[0], -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    mask=padding_mask.unsqueeze(axis=1).unsqueeze(axis=2).to("bool"),
                    value=fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)
        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )
        mask_pos_t = paddle.ones(shape=(x.shape[0], x.shape[1]))
        mask_pos_t = mask_pos_t.astype(dtype=x.dtype)
        if mask_pos is not None:
            mask_pos_t.masked_fill_(mask=mask_pos.to("bool"), value=0)
        x_norm = (
            x.astype(dtype="float32").norm(axis=-1) - math.sqrt(x.shape[-1])
        ).abs()
        mask_pos_t.masked_fill_(mask=x_norm <= 1, value=0)
        mask_pos_t = mask_pos_t.to("bool")
        if mask_pos_t.astype("bool").any():
            x_norm = x_norm[mask_pos_t].mean()
        else:
            x_norm = paddle.zeros(shape=[1])
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len)
            .transpose(perm=[0, 2, 3, 1])
            .contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .transpose(perm=[0, 2, 3, 1])
            .contiguous()
        )
        delta_pair_repr_norm = delta_pair_repr.astype(dtype="float32").norm(axis=-1)
        mask_pos_t = paddle.ones_like(x=delta_pair_repr_norm)
        if mask_pos is not None:
            mask_pos_t.masked_fill_(mask=mask_pos.unsqueeze(axis=1).to("bool"), value=0)
            mask_pos_t = mask_pos_t.transpose(perm=[0, 2, 1])
            mask_pos_t.masked_fill_(mask=mask_pos.unsqueeze(axis=1).to("bool"), value=0)
            mask_pos_t = mask_pos_t.transpose(perm=[0, 2, 1])
        delta_pair_repr_norm = (
            delta_pair_repr_norm - math.sqrt(delta_pair_repr.shape[-1])
        ).abs()
        mask_pos_t.masked_fill_(mask=delta_pair_repr_norm <= 1, value=0)
        mask_pos_t = mask_pos_t.to("bool")
        if mask_pos_t.astype("bool").any():
            delta_pair_repr_norm = delta_pair_repr_norm[mask_pos_t].mean()
        else:
            delta_pair_repr_norm = paddle.zeros(shape=[1])
        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)
        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm
