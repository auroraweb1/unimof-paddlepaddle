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

from .unimat import UniMatModel

logger = logging.getLogger(__name__)
MIN_MAX_KEY = {
    "hmof": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "CoRE_MAP": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "CoRE_MAP_LargeScale": {"pressure": [-4.0, 8.0], "temperature": [100, 500.0]},
    "CoRE_DB": {"pressure": [-4.0, 6.0], "temperature": [70, 90.0]},
    "EXP_ADS": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "EXP_ADS_hmof": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "CoRE_MAP_CH4": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "CoRE_MAP_CO2": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "CoRE_MAP_Ar": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "CoRE_MAP_Kr": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "CoRE_MAP_Xe": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "CoRE_MAP_O2": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
    "CoRE_MAP_N2": {"pressure": [-4.0, 6.0], "temperature": [100, 400.0]},
}


@register_model("unimof_v2_NoGasID")
class UniMOFV2Model_NoGasID(BaseUnicoreModel):
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--gas-attr-input-dim", type=int, default=6, help="size of gas feature"
        )
        parser.add_argument(
            "--hidden-dim", type=int, default=128, help="output dimension of embedding"
        )
        parser.add_argument(
            "--bins",
            type=int,
            default=32,
            help="number of bins for temperature and pressure",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            type=str,
            default="tanh",
            help="pooler activation function",
        )
        parser.add_argument(
            "--pooler-dropout", type=float, default=0.1, help="pooler dropout"
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.unimat = UniMatModel(self.args, dictionary)
        self.min_max_key = MIN_MAX_KEY[args.task_name]
        self.gas_embed = GasModel(self.args.gas_attr_input_dim, self.args.hidden_dim)
        self.env_embed = EnvModel(
            self.args.hidden_dim, self.args.bins, self.min_max_key
        )
        self.classifier = ClassificationHead(
            args.encoder_embed_dim + self.args.hidden_dim * 4,
            self.args.hidden_dim * 2,
            self.args.num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        gas_attr,
        pressure,
        temperature,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        **kwargs
    ):
        """Forward pass for the UniMofAbsorbModel model."""

        def get_dist_features(dist, et):
            n_node = dist.shape[-1]
            gbf_feature = self.unimat.gbf(dist, et)
            gbf_result = self.unimat.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.transpose(perm=[0, 3, 1, 2]).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        padding_mask = src_tokens.equal(y=self.unimat.padding_idx)
        mol_x = self.unimat.embed_tokens(src_tokens)
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        encoder_outputs = self.unimat.encoder(
            mol_x, padding_mask=padding_mask, attn_mask=graph_attn_bias
        )
        cls_repr = encoder_outputs[0][:, 0, :]
        gas_embed = self.gas_embed(gas_attr)
        env_embed = self.env_embed(pressure, temperature)
        rep = paddle.concat(x=[cls_repr, gas_embed, env_embed], axis=-1)
        logits = self.classifier(rep)
        return [logits]

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class GasModel(paddle.nn.Layer):
    def __init__(self, gas_attr_input_dim, gas_dim, gas_max_count=500):
        super().__init__()
        self.gas_attr_embed = NonLinearHead(gas_attr_input_dim, gas_dim, "relu")

    def forward(self, gas_attr):
        gas_attr = gas_attr.astype(dtype=self.gas_attr_embed.linear1.weight.dtype)
        gas_attr_embed = self.gas_attr_embed(gas_attr)
        gas_repr = paddle.concat(x=[gas_attr_embed], axis=-1)
        return gas_repr


class EnvModel(paddle.nn.Layer):
    def __init__(self, hidden_dim, bins=32, min_max_key=None):
        super().__init__()
        self.project = NonLinearHead(2, hidden_dim, "relu")
        self.bins = bins
        self.pressure_embed = Embedding(num_embeddings=bins, embedding_dim=hidden_dim)
        self.temperature_embed = Embedding(
            num_embeddings=bins, embedding_dim=hidden_dim
        )
        self.min_max_key = min_max_key

    def forward(self, pressure, temperature):
        pressure = pressure.astype(dtype=self.project.linear1.weight.dtype)
        temperature = temperature.astype(dtype=self.project.linear1.weight.dtype)
        pressure = paddle.clip(
            x=pressure,
            min=self.min_max_key["pressure"][0],
            max=self.min_max_key["pressure"][1],
        )
        temperature = paddle.clip(
            x=temperature,
            min=self.min_max_key["temperature"][0],
            max=self.min_max_key["temperature"][1],
        )
        pressure = (pressure - self.min_max_key["pressure"][0]) / (
            self.min_max_key["pressure"][1] - self.min_max_key["pressure"][0]
        )
        temperature = (temperature - self.min_max_key["temperature"][0]) / (
            self.min_max_key["temperature"][1] - self.min_max_key["temperature"][0]
        )
        env_project = paddle.concat(
            x=(pressure[:, None], temperature[:, None]), axis=-1
        )
        env_project = self.project(env_project)
        pressure_bin = paddle.floor(x=pressure * self.bins).to("int64")
        temperature_bin = paddle.floor(x=temperature * self.bins).to("int64")
        pressure_embed = self.pressure_embed(pressure_bin)
        temperature_embed = self.temperature_embed(temperature_bin)
        env_embed = paddle.concat(x=[pressure_embed, temperature_embed], axis=-1)
        env_repr = paddle.concat(x=[env_project, env_embed], axis=-1)
        return env_repr


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
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("unimof_v2_NoGasID", "unimof_v2_NoGasID")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.lattice_loss = getattr(args, "lattice_loss", -1.0)
    args.gas_attr_input_dim = getattr(args, "gas_attr_input_dim", 6)
    args.gas_dim = getattr(args, "hidden_dim", 128)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "relu")
