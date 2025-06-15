import paddle


class ModuleProxyWrapper(paddle.nn.Layer):
    """
    Wrap a DistributedDataParallel module and forward requests for missing
    attributes to the module wrapped by DDP (the twice-wrapped module).
    Also forward calls to :func:`state_dict` and :func:`load_state_dict`.

    Usage::

        module.xyz = "hello world"
        wrapped_module = DistributedDataParallel(module, **ddp_args)
        wrapped_module = ModuleProxyWrapper(wrapped_module)
        assert wrapped_module.xyz == "hello world"
        assert wrapped_module.state_dict().keys() == module.state_dict().keys()

    Args:
        module (nn.Module): module to wrap
    """

    def __init__(self, module: paddle.nn.Layer):
        super().__init__()
        assert hasattr(
            module, "module"
        ), "ModuleProxyWrapper expects input to wrap another module"
        self.module = module

    def __getattr__(self, name):
        """Forward missing attributes to twice-wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self.module, name)
            except AttributeError:
                return getattr(self.module.module, name)

    def state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        """Not Support auto convert *.state_dict, please judge whether it is Pytorch API and convert by yourself"""
        return self.module.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        """Not Support auto convert *.load_state_dict, please judge whether it is Pytorch API and convert by yourself"""
        return self.module.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def bfloat16(self):
        return self.module.module.astype(dtype="bfloat16")

    def half(self):
        return self.module.module.astype(dtype="float16")
