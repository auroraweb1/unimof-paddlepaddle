from typing import Dict, Optional

import paddle
from unicore import utils

from . import LayerNorm, SelfMultiheadAttention


class TransformerEncoderLayer(paddle.nn.Layer):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln=False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = SelfMultiheadAttention(
            self.embed_dim, attention_heads, dropout=attention_dropout
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = paddle.nn.Linear(
            in_features=self.embed_dim, out_features=ffn_embed_dim
        )
        self.fc2 = paddle.nn.Linear(
            in_features=ffn_embed_dim, out_features=self.embed_dim
        )
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln

    def forward(
        self,
        x: paddle.Tensor,
        attn_bias: Optional[paddle.Tensor] = None,
        padding_mask: Optional[paddle.Tensor] = None,
        return_attn: bool = False,
    ) -> paddle.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = paddle.nn.functional.dropout(
            x=x, p=self.activation_dropout, training=self.training
        )
        x = self.fc2(x)
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs
