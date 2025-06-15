import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
import logging
import math
from collections.abc import Collection
from typing import List

import paddle
from paddle_utils import *
from unicore.optim import UnicoreOptimizer, register_optimizer
from unicore.optim.fused_adam import get_fused_adam_class

logger = logging.getLogger(__name__)


@register_optimizer("adam")
class UnicoreAdam(UnicoreOptimizer):
    """Adam optimizer for unicore.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        fused_adam_cls = get_fused_adam_class()
        use_fused_adam = (
            not getattr(args, "use_old_adam", False)
            and fused_adam_cls is not None
            and paddle.device.cuda.device_count() >= 1
            and paddle.device.cuda.get_device_capability()[0] >= 7
        )
        if use_fused_adam:
            logger.info("using FusedAdam")
            self._optimizer = fused_adam_cls(params, **self.optimizer_config)
        else:
            self._optimizer = Adam(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument(
            "--adam-betas",
            default="(0.9, 0.999)",
            metavar="B",
            help="betas for Adam optimizer",
        )
        parser.add_argument(
            "--adam-eps",
            type=float,
            default=1e-08,
            metavar="D",
            help="epsilon for Adam optimizer",
        )
        parser.add_argument(
            "--weight-decay",
            "--wd",
            default=0.0,
            type=float,
            metavar="WD",
            help="weight decay",
        )

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.args.lr[0]
            if isinstance(self.args.lr, Collection)
            else self.args.lr,
            "betas": eval(self.args.adam_betas),
            "eps": self.args.adam_eps,
            "weight_decay": self.args.weight_decay,
        }


class Adam(paddle.optimizer.Optimizer):
    """Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(Adam, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {"float16", "bfloat16"}:
                    grad = grad.astype(dtype="float32")
                if grad.is_sparse():
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group.get("amsgrad", False)
                p_data_fp32 = p.data
                if p.data.dtype in {"float16", "bfloat16"}:
                    p_data_fp32 = p_data_fp32.astype(dtype="float32")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = paddle.zeros_like(x=p_data_fp32)
                    state["exp_avg_sq"] = paddle.zeros_like(x=p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = paddle.zeros_like(x=p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.multiply_(y=paddle.to_tensor(beta1)).add_(
                    y=paddle.to_tensor((1 - beta1) * grad)
                )
                exp_avg_sq.multiply_(y=paddle.to_tensor(beta2)).add_(
                    (1 - beta2) * grad * grad
                )
                if amsgrad:
                    paddle_max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(y=paddle.to_tensor(group["eps"]))
                else:
                    denom = exp_avg_sq.sqrt().add_(y=paddle.to_tensor(group["eps"]))
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        y=paddle.to_tensor(
                            -group["weight_decay"] * group["lr"] * p_data_fp32
                        )
                    )
                p_data_fp32.add_(-step_size * exp_avg / denom)
                if p.data.dtype in {"float16", "bfloat16"}:
                    p.data.copy_(p_data_fp32)
        return loss
