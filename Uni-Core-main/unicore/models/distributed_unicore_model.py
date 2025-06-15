import logging

import paddle
from unicore.distributed import (LegacyDistributedDataParallel,
                                 ModuleProxyWrapper)

logger = logging.getLogger(__name__)


def DistributedUnicoreModel(args, model, process_group, device):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): unicore args
        model (BaseUnicoreModel): model to wrap
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
        device: device to move model to
    """
    assert isinstance(model, paddle.nn.Layer)
    if args.ddp_backend in {"c10d", "pytorch_ddp"}:
        wrapped_model = paddle.DataParallel(
            model.to(device),
            find_unused_parameters=args.find_unused_parameters
        )
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {"apex"}:
        import apex

        wrapped_model = apex.parallel.DistributedDataParallel(module=model.to(device))
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {"no_c10d", "legacy_ddp"}:
        wrapped_model = LegacyDistributedDataParallel(
            module=model.to(device), buffer_size=2**28, process_group=process_group
        )
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    else:
        raise ValueError("Unknown --ddp-backend: " + args.ddp_backend)
    return wrapped_model
