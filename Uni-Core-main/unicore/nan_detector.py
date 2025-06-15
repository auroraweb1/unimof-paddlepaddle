import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
import logging

import paddle
from paddle_utils import *

logger = logging.getLogger(__name__)


class NanDetector:
    """
    Detects the first NaN or Inf in forward and/or backward pass and logs, together with the module name
    """

    def __init__(self, model, forward=True, backward=True):
        self.bhooks = []
        self.fhooks = []
        self.forward = forward
        self.backward = backward
        self.named_parameters = list(model.named_parameters())
        self.reset()
        for name, mod in model.named_sublayers(include_self=True):
            mod.__module_name = name
            self.add_hooks(mod)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        norm = {}
        gradients = {}
        for name, param in self.named_parameters:
            if param.grad is not None:
                grad_norm = paddle.linalg.norm(x=param.grad.data, p=2).astype("float32")
                norm[name] = grad_norm.item()
                if (
                    paddle.isnan(x=grad_norm).astype("bool").any()
                    or paddle.isinf(x=grad_norm).astype("bool").any()
                ):
                    gradients[name] = param.grad.data
        if len(gradients) > 0:
            logger.info("Detected nan/inf grad norm, dumping norms...")
            logger.info(f"norms: {norm}")
            logger.info(f"gradients: {gradients}")
        self.close()

    def add_hooks(self, module):
        if self.forward:
            self.fhooks.append(module.register_forward_post_hook(hook=self.fhook_fn))
        if self.backward:
            """Not Support auto convert *.register_backward_hook, please judge whether it is Pytorch API and convert by yourself"""
>>>>>>            self.bhooks.append(module.register_backward_hook(self.bhook_fn))

    def reset(self):
        self.has_printed_f = False
        self.has_printed_b = False

    def _detect(self, tensor, name, backward):
        err = None
        if paddle.is_floating_point(x=tensor) and tensor.size >= 2:
            with paddle.no_grad():
                if paddle.isnan(x=tensor).astype("bool").any():
                    err = "NaN"
                elif paddle.isinf(x=tensor).astype("bool").any():
                    err = "Inf"
        if err is not None:
            err = f"{err} detected in output of {name}, shape: {tuple(tensor.shape)}, {'backward' if backward else 'forward'}"
        return err

    def _apply(self, module, inp, x, backward):
        if paddle.is_tensor(x=x):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            err = self._detect(x, module.__module_name, backward)
            if err is not None:
                if paddle.is_tensor(x=inp) and not backward:
                    err += f" input max: {inp._max().item()}, input min: {inp._min().item()}"
                has_printed_attr = "has_printed_b" if backward else "has_printed_f"
                logger.warning(err)
                setattr(self, has_printed_attr, True)
        elif isinstance(x, dict):
            for v in x.values():
                self._apply(module, inp, v, backward)
        elif isinstance(x, list) or isinstance(x, tuple):
            for v in x:
                self._apply(module, inp, v, backward)

    def fhook_fn(self, module, inp, output):
        if not self.has_printed_f:
            self._apply(module, inp, output, backward=False)

    def bhook_fn(self, module, inp, output):
        if not self.has_printed_b:
            self._apply(module, inp, output, backward=True)

    def close(self):
        for hook in self.fhooks + self.bhooks:
            hook.remove()
