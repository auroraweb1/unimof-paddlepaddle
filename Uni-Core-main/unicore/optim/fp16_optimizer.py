import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
from collections import defaultdict

import paddle
from paddle_utils import *
from unicore import optim, utils

from .dynamic_loss_scaler import DynamicLossScaler


def separate_decay_params(args, params):
    if args.weight_decay <= 0:
        return [{"params": [p for _, p in params if not p.stop_gradient]}]
    no_wd = (
        set(args.no_weight_decay_names.split(","))
        if args.no_weight_decay_names
        else set()
    )

    def skip_decay(name, p):
        return name.endswith(".bias") or p.ndim == 1 or any(nd in name for nd in no_wd)

    decay_params = []
    no_decay_params = []
    for name, p in params:
        if not not p.stop_gradient:
            continue
        elif skip_decay(name, p):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    ret = []
    if len(decay_params) > 0:
        ret.append({"params": decay_params})
    if len(no_decay_params) > 0:
        ret.append({"params": no_decay_params, "weight_decay": 0.0})
    return ret


def check_param_device(params):
    if len(params) <= 0:
        return True
    device = params[0].place
    for i in range(1, len(params)):
        assert device == params[i].place


def pad_numel(numel, multiplier=2):
    return (numel + multiplier - 1) // multiplier * multiplier


def flatten_orders(params):
    dtype_grouped_params = {}
    ordered_dtype = []
    total_param_size = 0
    for p in params:
        if p.dtype not in dtype_grouped_params:
            dtype_grouped_params[p.dtype] = []
            ordered_dtype.append(p.dtype)
        dtype_grouped_params[p.dtype].append(p)
        total_param_size += pad_numel(p.data.numel())
    return dtype_grouped_params, ordered_dtype, total_param_size


@paddle.no_grad()
def flatten_parameters(params):
    dtype_grouped_params, ordered_dtype, _ = flatten_orders(params)
    flatten_params = {}
    for dtype in ordered_dtype:
        cur_params = dtype_grouped_params[dtype]
        total_param_size = sum(pad_numel(p.data.numel()) for p in cur_params)
        flatten_params[dtype] = paddle.zeros(
            shape=total_param_size, dtype=cur_params[0].new(0).astype(dtype).dtype
        )
        offset = 0
        for p in cur_params:
            numel = p.data.numel()
            paddle.assign(
                p.data.view(-1), output=flatten_params[dtype][offset : offset + numel]
            )
            p.data = (
                flatten_params[dtype]
                .data[offset : offset + numel]
                .view(*tuple(p.shape))
            )
            offset += pad_numel(numel)
        flatten_params[dtype] = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=flatten_params[dtype]
        )
        flatten_params[dtype].grad = flatten_params[dtype].data.new(total_param_size)
        offset = 0
        for p in cur_params:
            numel = p.data.numel()
            p.grad = (
                flatten_params[dtype]
                .grad[offset : offset + numel]
                .view(*tuple(p.shape))
            )
            offset += pad_numel(numel)
    paddle.device.cuda.empty_cache()
    return [flatten_params[dtype] for dtype in ordered_dtype]


@paddle.no_grad()
def flatten_parameters_fp32(params, set_to_param=False, set_grad=True):
    dtype_grouped_params, ordered_dtype, total_param_size = flatten_orders(params)
    flatten_params = paddle.zeros(shape=total_param_size, dtype="float32")
    offset = 0
    for dtype in ordered_dtype:
        cur_params = dtype_grouped_params[dtype]
        for p in cur_params:
            numel = p.data.numel()
            paddle.assign(
                p.data.view(-1), output=flatten_params[offset : offset + numel]
            )
            if set_to_param:
                p.data = flatten_params.data[offset : offset + numel].view(
                    *tuple(p.shape)
                )
                p.grad = None
            offset += pad_numel(numel)
    flatten_params = paddle.base.framework.EagerParamBase.from_tensor(
        tensor=flatten_params
    )
    if set_grad:
        flatten_params.grad = paddle.zeros_like(x=flatten_params)
    paddle.device.cuda.empty_cache()
    return flatten_params


def get_fp16_params(args, params):
    param_group = separate_decay_params(args, params)
    fp16_group = []
    fp32_group = []
    for param_dict in param_group:
        params = param_dict["params"]
        check_param_device(params)
        fp16_params = flatten_parameters(params)
        fp32_params = flatten_parameters_fp32(params)
        fp16_group.append({"params": fp16_params})
        param_dict["params"] = [fp32_params]
        fp32_group.append(param_dict)
    return fp16_group, fp32_group


class _FP16OptimizerMixin(object):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self._multiply_factor = 1.0
        self.bf16_sr = getattr(args, "bf16_sr", False)

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.fp32_optimizer.state_dict()
        if self.scaler is not None:
            state_dict["loss_scale"] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.
        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        if "loss_scale" in state_dict and self.scaler is not None:
            self.scaler.loss_scale = state_dict["loss_scale"]
        self.fp32_optimizer.set_state_dict(state_dict=state_dict)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.
        Compared to :func:`unicore.optim.UnicoreOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
        if self.scaler is not None:
            loss = self.scaler.scale(loss)
        loss.backward()
        self._needs_sync = True

    def _sync_fp16_grads_to_fp32(self):
        with paddle.no_grad():
            if self._needs_sync:
                for gid in range(len(self.fp16_params)):
                    offset = 0
                    for p in self.fp16_params[gid]["params"]:
                        numel = p.size
                        paddle.assign(
                            p.grad.data.view(-1),
                            output=self.fp32_params[gid]["params"][0].grad.data[
                                offset : offset + numel
                            ],
                        )
                        offset += pad_numel(numel)
                    self._needs_sync = False

    def _add_fp16_grads_to_fp32(self, mul=0.0):
        with paddle.no_grad():
            for gid in range(len(self.fp16_params)):
                offset = 0
                for p in self.fp16_params[gid]["params"]:
                    numel = p.size
                    self.fp32_params[gid]["params"][0].grad.data[
                        offset : offset + numel
                    ] += mul * p.grad.data.float().view(-1)
                    p.grad.zero_()
                    offset += pad_numel(numel)
                self._needs_sync = False

    def _sync_fp32_params_to_fp16(self):
        for gid in range(len(self.fp16_params)):
            offset = 0
            for p in self.fp16_params[gid]["params"]:
                numel = p.size
                u = (
                    self.fp32_params[gid]["params"][0]
                    .data[offset : offset + numel]
                    .view_as(other=p.data)
                )
                if self.bf16_sr and p.dtype == "bfloat16":
                    utils.fp32_to_bf16_sr(u, p)
                else:
                    p.data.copy_(u)
                offset += pad_numel(numel)

    def _unscale_grads(self):
        self._sync_fp16_grads_to_fp32()
        if paddle.is_tensor(x=self._multiply_factor) or self._multiply_factor != 1.0:
            self.fp32_optimizer.multiply_grads(self._multiply_factor)
            self._multiply_factor = 1.0

    def multiply_grads(self, c):
        """Multiplies grads by a constant ``c``."""
        if self._needs_sync:
            self._multiply_factor *= c
        else:
            self.fp32_optimizer.multiply_grads(c)

    def per_sample_clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""
        if max_norm <= 0.0:
            return 0.0
        all_fp16_params = defaultdict(list)
        for p in self.fp16_params:
            all_fp16_params.extend(p["params"])
        grad_norm = self._multiply_factor * utils.clip_grad_norm_(
            all_fp16_params, 0, aggregate_norm_fn
        )
        if grad_norm > max_norm > 0.0:
            clip_coef = max_norm / (grad_norm + 1e-06)
        else:
            clip_coef = 1.0
        self._add_fp16_grads_to_fp32(mul=clip_coef)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm and updates dynamic loss scaler."""
        self._sync_fp16_grads_to_fp32()
        grad_norm = self._multiply_factor * self.fp32_optimizer.clip_grad_norm(
            0, aggregate_norm_fn=aggregate_norm_fn
        )
        if self.scaler is not None:
            if grad_norm > max_norm > 0.0:
                self._multiply_factor *= max_norm / grad_norm
            self.scaler.check_overflow(grad_norm)
        elif max_norm > 0.0:
            clip_coef = (max_norm / (grad_norm + 1e-06)).clip_(max=1)
            self._multiply_factor *= clip_coef
        return grad_norm

    def step(self, closure=None, groups=None):
        """Performs a single optimization step."""
        self._sync_fp16_grads_to_fp32()
        if getattr(self, "supports_step_with_scale", False):
            self.fp32_optimizer.step(
                closure, scale=1.0 / self._multiply_factor, groups=groups
            )
        else:
            self._unscale_grads()
            self.fp32_optimizer.step(closure, groups=groups)
        if self.scaler is not None:
            self.scaler.update()
        self._sync_fp32_params_to_fp16()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""

        def zero(group):
            for x in group:
                for p in x["params"]:
                    p.grad.zero_()

        zero(self.fp16_params)
        zero(self.fp32_params)
        self._needs_sync = False
        if self.scaler is not None:
            self._multiply_factor = 1.0 / float(self.scaler.loss_scale)
        else:
            self._multiply_factor = 1.0


class FP16Optimizer(_FP16OptimizerMixin, optim.UnicoreOptimizer):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.
    """

    def __init__(self, args, params, fp32_optimizer, fp32_params, **kwargs):
        super().__init__(args)
        self.fp16_params = params
        self.fp32_optimizer = fp32_optimizer
        self.fp32_params = fp32_params
        self.allreduce_fp32_grad = getattr(args, "allreduce_fp32_grad", False)
        if getattr(args, "fp16_scale_window", None) is None:
            if len(args.update_freq) > 1:
                raise ValueError(
                    "--fp16-scale-window must be given explicitly when using a custom --update-freq schedule"
                )
            data_parallel_size = int(args.distributed_world_size)
            scale_window = int(2**14 / data_parallel_size / args.update_freq[0])
        else:
            scale_window = args.fp16_scale_window
        if not getattr(args, "bf16", False):
            self.scaler = DynamicLossScaler(
                init_scale=args.fp16_init_scale,
                scale_window=scale_window,
                tolerance=args.fp16_scale_tolerance,
                threshold=args.threshold_loss_scale,
                min_loss_scale=args.min_loss_scale,
            )
        else:
            self.scaler = None

    @classmethod
    def build_optimizer(cls, args, params, **kwargs):
        """
        Args:
            args : unicore args
            params (iterable): iterable of parameters to optimize
        """
        flatten = not getattr(args, "fp16_no_flatten_grads", False)
        assert flatten
        fp16_group, fp32_group = get_fp16_params(args, params)
        fp32_optimizer = optim.build_optimizer(args, fp32_group, separate=False)
        return cls(args, fp16_group, fp32_optimizer, fp32_group, **kwargs)

    @property
    def optimizer(self):
        return self.fp32_optimizer.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.fp32_optimizer.optimizer = optimizer

    @property
    def lr_scheduler(self):
        return getattr(self.fp32_optimizer, "lr_scheduler", None)

    @property
    def optimizer_config(self):
        return self.fp32_optimizer.optimizer_config

    def get_lr(self):
        return self.fp32_optimizer.get_lr()

    def set_lr(self, lr):
        self.fp32_optimizer.set_lr(lr)

    def all_reduce_grads(self, module):
        if self.allreduce_fp32_grad and hasattr(module, "all_reduce_params"):
            self._sync_fp16_grads_to_fp32()
            with paddle.no_grad():
                params = [x["params"][0] for x in self.fp32_params]
                module.all_reduce_params(params)
        else:
            self.fp32_optimizer.all_reduce_grads(module)

    @property
    def supports_flat_params(self):
        return self.fp32_optimizer.supports_flat_params
