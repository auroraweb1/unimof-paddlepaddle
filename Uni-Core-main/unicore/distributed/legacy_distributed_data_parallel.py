import paddle

"""
A modified version of the legacy DistributedDataParallel module that uses c10d
communication primitives. This version is simpler than the latest PyTorch
version and is useful for debugging. Notably it does not overlap gradient
communication with the backward pass, which makes it slower but more robust
than the PyTorch version.

This version also supports the *no_sync* context manager, which allows faster
training with `--update-freq`.
"""
from collections import OrderedDict
from contextlib import contextmanager

from unicore.distributed import utils


class LegacyDistributedDataParallel(paddle.nn.Layer):
    """Implements distributed data parallelism at the module level.

    A simplified version of :class:`torch.nn.parallel.DistributedDataParallel`.
    This version uses a c10d process group for communication and does not
    broadcast buffers.

    Args:
        module (~torch.nn.Module): module to be parallelized
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
        buffer_size (int, optional): number of elements to buffer before
            performing all-reduce (default: 256M).
    """

    def __init__(self, module, process_group, buffer_size=2**28):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.world_size = utils.get_world_size(self.process_group)
        self.buffer_size = min(buffer_size, sum(p.size for p in module.parameters()))
        self.buffer = None
        self.accumulate_grads = False
        paramlists = OrderedDict()
        for param in self.module.parameters():
            device = param.place
            if paramlists.get(device) is None:
                paramlists[device] = []
            paramlists[device] += [param]
        self.per_device_params = list(paramlists.values())

    @contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        old_accumulate_grads = self.accumulate_grads
        self.accumulate_grads = True
        yield
        self.accumulate_grads = old_accumulate_grads

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def all_reduce_params(self, params):
        if self.accumulate_grads:
            return
        buffer = self.buffer
        nonzero_buffer = False
        if len(params) > 1:
            offset = 0
            for p in params:
                sz = p.size
                if p.grad is not None:
                    paddle.assign(
                        p.grad.data.view(-1), output=buffer[offset : offset + sz]
                    )
                    nonzero_buffer = True
                else:
                    buffer[offset : offset + sz].zero_()
                offset += sz
        else:
            p = params[0]
            if p.grad is not None:
                buffer = p.grad.data
                nonzero_buffer = True
            elif p.size <= self.buffer.size:
                buffer = buffer[: p.size]
                buffer.zero_()
            else:
                buffer = paddle.zeros_like(x=p)
        if nonzero_buffer:
            buffer.divide_(y=paddle.to_tensor(self.world_size))
        utils.all_reduce(buffer, self.process_group)
        offset = 0
        for p in params:
            sz = p.size
            if p.grad is not None:
                p.grad.data.copy_(buffer[offset : offset + sz].view_as(other=p))
            else:
                p.grad = buffer[offset : offset + sz].view_as(other=p).clone()
            offset += sz

    def all_reduce_grads(self):
        """
        This function must be called explicitly after backward to reduce
        gradients. There is no automatic hook like c10d.
        """

        def reduction_fn():
            if self.accumulate_grads:
                return
            if self.buffer is None:
                self.buffer = next(self.module.parameters()).new(self.buffer_size)
            for params in self.per_device_params:
                offset = 0
                buffered_params = []
                for param in params:
                    if not not param.stop_gradient:
                        continue
                    if param.grad is None:
                        param.grad = paddle.zeros_like(x=param)
                    if hasattr(param, "expert"):
                        continue
                    if param.grad.requires_grad:
                        raise RuntimeError(
                            "DistributedDataParallel only works with gradients that don't require grad"
                        )
                    sz = param.size
                    if sz > self.buffer.size:
                        self.all_reduce_params([param])
                    else:
                        if offset + sz > self.buffer.size:
                            self.all_reduce_params(buffered_params)
                            offset = 0
                            buffered_params.clear()
                        buffered_params.append(param)
                        offset += sz
                if len(buffered_params) > 0:
                    self.all_reduce_params(buffered_params)

        reduction_fn()
