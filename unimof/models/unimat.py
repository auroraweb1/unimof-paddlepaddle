import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
import logging
from typing import Any, Dict, List

import paddle
from paddle_utils import *
from unicore import utils
from unicore.models import (BaseUnicoreModel, register_model,
                            register_model_architecture)
from unicore.modules import LayerNorm, init_bert_params

from .transformer_encoder_with_pair import TransformerEncoderWithPair

logger = logging.getLogger(__name__)


@register_model("unimat")
class UniMatModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss", type=float, metavar="D", help="mask loss ratio"
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss", type=float, metavar="D", help="x norm loss ratio"
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--lattice-loss", type=float, metavar="D", help="lattice loss ratio"
        )
        parser.add_argument(
            "--gaussian-kernel",
            action="store_true",
            help="use gaussian kernel for distance map",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=args.encoder_embed_dim,
            padding_idx=self.padding_idx,
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=self.embed_tokens.weight,
            )
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        if "gaussian_kernel" in args and args.gaussian_kernel:
            self.gbf = GaussianLayer(K, n_edge_type)
        else:
            self.gbf = NumericalEmbed(K, n_edge_type)
        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )
        if args.lattice_loss > 0:
            self.lattice_head = ClassificationHead(
                input_dim=args.encoder_embed_dim,
                inner_dim=args.encoder_embed_dim,
                num_classes=3,
                activation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        self.classification_heads = paddle.nn.LayerDict()
        self.node_classification_heads = paddle.nn.LayerDict()
        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True
        padding_mask = src_tokens.equal(y=self.padding_idx)
        if not padding_mask.astype("bool").any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.shape[-1]
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.transpose(perm=[0, 3, 1, 2]).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        encoder_distance = None
        encoder_coord = None
        lattice = None
        logits = None
        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = (
                        paddle.sum(x=1 - padding_mask.astype(dtype=x.dtype), axis=1) - 1
                    ).view(-1, 1, 1, 1)
                else:
                    atom_num = tuple(src_coord.shape)[1] - 1
                delta_pos = coords_emb.unsqueeze(axis=1) - coords_emb.unsqueeze(axis=2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = paddle.sum(x=coord_update, axis=2)
                encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(encoder_pair_rep)
            if self.args.lattice_loss > 0:
                lattice = self.lattice_head(encoder_rep)
        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](encoder_rep)
        elif features_only and (
            self.classification_heads or self.node_classification_heads
        ):
            logits = {}
            for name, head in self.node_classification_heads.items():
                logits[name] = head(encoder_rep)
            for name, head in self.classification_heads.items():
                logits[name] = head(encoder_rep)
        return (
            logits,
            encoder_distance,
            encoder_coord,
            lattice,
            x_norm,
            delta_encoder_pair_rep_norm,
        )

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def register_node_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = NodeClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(paddle.nn.Layer):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = paddle.nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        if weight is None:
            weight = paddle.nn.Linear(
                in_features=embed_dim, out_features=output_dim, bias_attr=False
            ).weight
        self.weight = weight
        self.bias = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=output_dim)
        )

    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = paddle.nn.functional.linear(x=x, weight=self.weight.T) + self.bias
        return x


class ClassificationHead(paddle.nn.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(
        self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout
    ):
        super().__init__()
        self.dense = paddle.nn.Linear(in_features=input_dim, out_features=inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = paddle.nn.Dropout(p=pooler_dropout)
        self.out_proj = paddle.nn.Linear(
            in_features=inner_dim, out_features=num_classes
        )

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NodeClassificationHead(paddle.nn.Layer):
    """Head for node-level classification tasks."""

    def __init__(
        self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout
    ):
        super().__init__()
        self.dense = paddle.nn.Linear(in_features=input_dim, out_features=inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = paddle.nn.Dropout(p=pooler_dropout)
        self.out_proj = paddle.nn.Linear(
            in_features=inner_dim, out_features=num_classes
        )

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(paddle.nn.Layer):
    """Head for simple classification tasks."""

    def __init__(self, input_dim, out_dim, activation_fn, hidden=None):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = paddle.nn.Linear(in_features=input_dim, out_features=hidden)
        self.linear2 = paddle.nn.Linear(in_features=hidden, out_features=out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(paddle.nn.Layer):
    def __init__(self, heads, activation_fn):
        super().__init__()
        self.dense = paddle.nn.Linear(in_features=heads, out_features=heads)
        self.layer_norm = paddle.nn.LayerNorm(heads)
        self.out_proj = paddle.nn.Linear(in_features=heads, out_features=1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = tuple(x.shape)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(perm=dim2perm(x.ndim, -1, -2))) * 0.5
        return x


@paddle.jit.to_static
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return paddle.exp(x=-0.5 * ((x - mean) / std) ** 2) / (a * std)


class GaussianLayer(paddle.nn.Layer):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = Embedding(num_embeddings=1, embedding_dim=K)
        self.stds = Embedding(num_embeddings=1, embedding_dim=K)
        self.mul = Embedding(num_embeddings=edge_types, embedding_dim=1)
        self.bias = Embedding(num_embeddings=edge_types, embedding_dim=1)
        init_Uniform = paddle.nn.initializer.Uniform(low=0, high=3)
        init_Uniform(self.means.weight)
        init_Uniform = paddle.nn.initializer.Uniform(low=0, high=3)
        init_Uniform(self.stds.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.bias.weight)
        init_Constant = paddle.nn.initializer.Constant(value=1)
        init_Constant(self.mul.weight)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).astype(dtype=x.dtype)
        bias = self.bias(edge_type).astype(dtype=x.dtype)
        x = mul * x.unsqueeze(axis=-1) + bias
        x = x.expand(shape=[-1, -1, -1, self.K])
        mean = self.means.weight.astype(dtype="float32").view(-1)
        std = self.stds.weight.astype(dtype="float32").view(-1).abs() + 1e-05
        return gaussian(x.astype(dtype="float32"), mean, std).astype(
            dtype=self.means.weight.dtype
        )


class NumericalEmbed(paddle.nn.Layer):
    def __init__(self, K=128, edge_types=1024, activation_fn="gelu"):
        super().__init__()
        self.K = K
        self.mul = Embedding(num_embeddings=edge_types, embedding_dim=1)
        self.bias = Embedding(num_embeddings=edge_types, embedding_dim=1)
        self.w_edge = Embedding(num_embeddings=edge_types, embedding_dim=K)
        self.proj = NonLinearHead(1, K, activation_fn, hidden=2 * K)
        self.ln = LayerNorm(K)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.bias.weight)
        init_Constant = paddle.nn.initializer.Constant(value=1)
        init_Constant(self.mul.weight)
        init_KaimingNormal = paddle.nn.initializer.KaimingNormal(
            nonlinearity="leaky_relu"
        )
        init_KaimingNormal(self.w_edge.weight)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).astype(dtype=x.dtype)
        bias = self.bias(edge_type).astype(dtype=x.dtype)
        w_edge = self.w_edge(edge_type).astype(dtype=x.dtype)
        edge_emb = w_edge * paddle.nn.functional.sigmoid(
            x=mul * x.unsqueeze(axis=-1) + bias
        )
        edge_proj = x.unsqueeze(axis=-1).astype(dtype=self.mul.weight.dtype)
        edge_proj = self.proj(edge_proj)
        edge_proj = self.ln(edge_proj)
        h = edge_proj + edge_emb
        h = h.astype(dtype=self.mul.weight.dtype)
        return h


@register_model_architecture("unimat", "unimat")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.lattice_loss = getattr(args, "lattice_loss", -1.0)


@register_model_architecture("unimat", "unimat_base")
def unimol_base_architecture(args):
    base_architecture(args)
