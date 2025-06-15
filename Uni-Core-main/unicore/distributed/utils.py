import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
import datetime
import io
import logging
import os
import pickle
import random
import socket
import struct
import subprocess
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import paddle
from paddle_utils import *

logger = logging.getLogger(__name__)


def is_master(args):
    return args.distributed_rank == 0


def infer_init_method(args, force_distributed=False):
    if args.distributed_init_method is not None:
        return
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        _infer_torch_distributed_launch_init(args)
    elif args.distributed_port > 0:
        _infer_slurm_init(args)
    elif args.distributed_world_size > 1 or force_distributed:
        _infer_single_node_init(args)
    elif not args.distributed_no_spawn:
        args.distributed_num_procs = min(
            paddle.device.cuda.device_count(), args.distributed_world_size
        )


def _infer_torch_distributed_launch_init(args):
    args.distributed_init_method = "env://"
    args.distributed_world_size = int(os.environ["WORLD_SIZE"])
    args.distributed_rank = int(os.environ["RANK"])
    args.distributed_no_spawn = True


def _infer_slurm_init(args):
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    if node_list is not None:
        try:
            hostnames = subprocess.check_output(
                ["scontrol", "show", "hostnames", node_list]
            )
            args.distributed_init_method = "tcp://{host}:{port}".format(
                host=hostnames.split()[0].decode("utf-8"), port=args.distributed_port
            )
            nnodes = int(os.environ.get("SLURM_NNODES"))
            ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
            if ntasks_per_node is not None:
                ntasks_per_node = int(ntasks_per_node)
            else:
                ntasks = int(os.environ.get("SLURM_NTASKS"))
                nnodes = int(os.environ.get("SLURM_NNODES"))
                assert ntasks % nnodes == 0
                ntasks_per_node = int(ntasks / nnodes)
            if ntasks_per_node == 1:
                gpus_per_node = paddle.device.cuda.device_count()
                node_id = int(os.environ.get("SLURM_NODEID"))
                args.distributed_rank = node_id * gpus_per_node
                args.distributed_world_size = nnodes * gpus_per_node
            else:
                assert ntasks_per_node == args.distributed_world_size // nnodes
                args.distributed_no_spawn = True
                args.distributed_rank = int(os.environ.get("SLURM_PROCID"))
                args.device_id = int(os.environ.get("SLURM_LOCALID"))
        except subprocess.CalledProcessError as e:
            raise e
        except FileNotFoundError:
            pass


def _infer_single_node_init(args):
    assert (
        args.distributed_world_size <= paddle.device.cuda.device_count()
    ), f"world size is {args.distributed_world_size} but have {paddle.device.cuda.device_count()} available devices"
    port = random.randint(10000, 20000)
    args.distributed_init_method = "tcp://localhost:{port}".format(port=port)


def distributed_init(args):
    if paddle.distributed.is_available() and paddle.distributed.is_initialized():
        warnings.warn("Distributed is already initialized, cannot initialize twice!")
    else:
        logger.info(
            "distributed init (rank {}): {}".format(
                args.distributed_rank, args.distributed_init_method
            )
        )
        paddle.distributed.init_parallel_env()
        logger.info(
            "initialized host {} as rank {}".format(
                socket.gethostname(), args.distributed_rank
            )
        )
        if paddle.device.cuda.device_count() >= 1:
            paddle.distributed.all_reduce(tensor=paddle.zeros(shape=[1]).cuda())
    args.distributed_rank = paddle.distributed.get_rank()
    if is_master(args):
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    return args.distributed_rank


def distributed_main(i, main, args, kwargs):
    args.device_id = i
    if paddle.device.cuda.device_count() >= 1 and not args.cpu:
        paddle.device.set_device(device=device2str(args.device_id))
    if args.distributed_rank is None:
        args.distributed_rank = kwargs.pop("start_rank", 0) + i
    args.distributed_rank = distributed_init(args)
    after_distributed_init_fn = kwargs.pop("after_distributed_init_fn", None)
    if after_distributed_init_fn:
        args = after_distributed_init_fn(args)
    main(args, **kwargs)
    if paddle.distributed.is_initialized():
        paddle.distributed.barrier(group=get_global_group())


def call_main(args, main, **kwargs):
    if args.distributed_init_method is None:
        infer_init_method(args)
    if args.distributed_init_method is not None:
        if not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None
            kwargs["start_rank"] = start_rank
            paddle.distributed.spawn(
                func=distributed_main,
                args=(main, args, kwargs),
                nprocs=min(
                    paddle.device.cuda.device_count(), args.distributed_world_size
                ),
                join=True,
            )
        else:
            distributed_main(int(os.environ["LOCAL_RANK"]), main, args, kwargs)
    else:
        main(args, **kwargs)


def get_rank(group):
    return paddle.distributed.get_rank(group=group)


def get_world_size(group):
    if paddle.distributed.is_initialized():
        return paddle.distributed.get_world_size(group=group)
    else:
        return 1


def get_global_group():
    return None


def get_global_rank():
    if paddle.distributed.is_initialized():
        return paddle.distributed.get_rank()
    else:
        return 0


def get_global_world_size():
    if paddle.distributed.is_initialized():
        return paddle.distributed.get_world_size()
    else:
        return 1


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    return get_global_group()


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_rank(get_data_parallel_group())


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_world_size(get_data_parallel_group())


def all_reduce(tensor, group, op="sum"):
    if op == "sum":
        op = paddle.distributed.ReduceOp.SUM
    elif op == "max":
        op = paddle.distributed.ReduceOp.MAX
    else:
        raise NotImplementedError
    paddle.distributed.all_reduce(tensor=tensor, op=op, group=group)
    return tensor


def broadcast(tensor, src, group):
    paddle.distributed.broadcast(tensor=tensor, src=src, group=group)


def all_to_all(tensor, group):
    """Perform an all-to-all operation on a 1D Tensor."""
    assert tensor.dim() == 1
    split_count = get_world_size(group=group)
    assert tensor.size % split_count == 0
    output = paddle.zeros_like(x=tensor)
    paddle.distributed.all_to_all([output], [tensor], group=group)
    return output


def all_gather(tensor, group, return_tensor=False):
    """Perform an all-gather operation."""
    world_size = get_world_size(group=group)
    rank = get_rank(group=group)
    tensor_list = [
        (tensor if i == rank else paddle.empty_like(x=tensor))
        for i in range(world_size)
    ]
    paddle.distributed.all_gather(tensor_list=tensor_list, tensor=tensor, group=group)
    if return_tensor:
        return paddle.stack(x=tensor_list, axis=0)
    else:
        return tensor_list


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable and any CUDA tensors will be moved
    to CPU and returned on CPU as well.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    from unicore import utils

    if group is None:
        group = get_global_group()
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)
    buffer_size = max_size * world_size
    if (
        not hasattr(all_gather_list, "_buffer")
        or all_gather_list._buffer.numel() < buffer_size
    ):
        all_gather_list._buffer = paddle.to_tensor(
            data=[0] * buffer_size, dtype="uint8", place="gpu"
        )
        all_gather_list._cpu_buffer = paddle.to_tensor(
            data=max_size, dtype="uint8"
        ).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer
    data = utils.move_to_cpu(data)
    enc = pickle.dumps(data)
    enc_size = len(enc)
    header_size = 4
    size = header_size + enc_size
    if size > max_size:
        raise ValueError(
            "encoded data size ({}) exceeds max_size ({})".format(size, max_size)
        )
    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = paddle.to_tensor(data=list(header + enc), dtype="uint8")
    start = rank * max_size
    paddle.assign(cpu_buffer[:size], output=buffer[start : start + size])
    all_reduce(buffer, group=group)
    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            (enc_size,) = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(
                    pickle.loads(
                        bytes(out_buffer[header_size : header_size + enc_size].tolist())
                    )
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all workers to enter the function together, so this error usually indicates that the workers have fallen out of sync somehow. Workers can fall out of sync if one of them runs out of memory, or if there are other conditions in your training script that can cause one worker to finish an epoch while other workers are still iterating over their portions of the data. Try rerunning with --ddp-backend=legacy_ddp and see if that helps."
        )


def all_reduce_dict(data: Mapping[str, Any], device, group) -> Dict[str, Any]:
    """
    AllReduce a dictionary of values across workers. We separately
    reduce items that are already on the device and items on CPU for
    better performance.

    Args:
        data (Mapping[str, Any]): dictionary of data to all-reduce, but
            cannot be a nested dictionary
        device (torch.device): device for the reduction
        group: group of the collective
    """
    data_keys = list(data.keys())
    cpu_data = OrderedDict()
    device_data = OrderedDict()
    for k in data_keys:
        t = data[k]
        if not paddle.is_tensor(x=t):
            cpu_data[k] = paddle.to_tensor(data=t, dtype="float64")
        elif t.device.type != device.type:
            cpu_data[k] = t.to(dtype="float64")
        else:
            device_data[k] = t.to(dtype="float64")

    def _all_reduce_dict(data: OrderedDict):
        if len(data) == 0:
            return data
        buf = paddle.concat(x=[t.view(-1) for t in data.values()]).to(device=device)
        all_reduce(buf, group=group)
        split_buf = paddle_split(x=buf, num_or_sections=[t.size for t in data.values()])
        reduced_data = [
            t.view_as(other=orig) for t, orig in zip(split_buf, data.values())
        ]
        return OrderedDict(zip(data.keys(), reduced_data))

    cpu_data = _all_reduce_dict(cpu_data)
    device_data = _all_reduce_dict(device_data)

    def get_from_stack(key):
        if key in cpu_data:
            return cpu_data[key]
        elif key in device_data:
            return device_data[key]
        raise KeyError

    return OrderedDict([(key, get_from_stack(key)) for key in data_keys])


@dataclass
class _TensorPlaceholder:
    index: int


def broadcast_tensors(
    tensors: Optional[List[paddle.Tensor]],
    src_rank: int,
    group: object,
    dist_device: Optional[str] = None,
) -> List[paddle.Tensor]:
    """
    Broadcasts a list of tensors without other (non-src) ranks needing to know
    the dtypes/shapes of the tensors.
    """
    if dist_device is None:
        if paddle.distributed.get_backend(group=group) == "nccl":
            dist_device = device2str("cuda")
        else:
            dist_device = device2str("cpu")
    is_src_rank = get_rank(group) == src_rank
    if is_src_rank:
        metadata = [
            {"size": tuple(t.shape), "dtype": t.dtype, "device": t.place}
            for t in tensors
        ]
        metadata = _broadcast_object_slow(metadata, src_rank, group, dist_device)
    else:
        metadata = _broadcast_object_slow(None, src_rank, group, dist_device)
    out_tensors = []
    for i, meta in enumerate(metadata):
        if is_src_rank:
            tensor = tensors[i]
            broadcast(tensors[i].to(dist_device), src=src_rank, group=group)
        else:
            tensor = paddle.zeros(shape=[meta[size].size], dtype=meta["dtype"])
            broadcast(tensor, src=src_rank, group=group)
        tensor = tensor.view(meta["size"]).to(meta["device"])
        out_tensors.append(tensor)
    return out_tensors


def broadcast_object(
    obj: Any, src_rank: int, group: object, dist_device: Optional[str] = None
) -> Any:
    """Broadcast an arbitrary Python object to other workers."""
    if dist_device is None:
        if paddle.distributed.get_backend(group=group) == "nccl":
            dist_device = device2str("cuda")
        else:
            dist_device = device2str("cpu")
    if get_rank(group) == src_rank:
        tensors = []
        obj = _split_tensors_from_obj(obj, tensors)
        obj = _broadcast_object_slow(obj, src_rank, group, dist_device)
        tensors = broadcast_tensors(tensors, src_rank, group, dist_device)
    else:
        obj = _broadcast_object_slow(None, src_rank, group, dist_device)
        tensors = broadcast_tensors(None, src_rank, group, dist_device)
    return _put_tensors_in_obj(obj, tensors)


def _broadcast_object_slow(
    obj: Any, src_rank: int, group: object, dist_device: str
) -> Any:
    if get_rank(group) == src_rank:
        buffer = io.BytesIO()
        paddle.save(obj=obj, path=buffer)
        buffer = paddle.to_tensor(data=buffer.getbuffer(), dtype="uint8").to(
            dist_device
        )
        length = paddle.to_tensor(data=[len(buffer)], dtype="int64").to(dist_device)
        broadcast(length, src=src_rank, group=group)
        broadcast(buffer, src=src_rank, group=group)
    else:
        length = paddle.to_tensor(data=[0], dtype="int64").to(dist_device)
        broadcast(length, src=src_rank, group=group)
        buffer = paddle.to_tensor(data=int(length.item()), dtype="uint8").to(
            dist_device
        )
        broadcast(buffer, src=src_rank, group=group)
        buffer = io.BytesIO(buffer.cpu().numpy())
        obj = paddle.load(path=str(buffer))
    return obj


def _split_tensors_from_obj(obj: Any, tensors: List[paddle.Tensor]) -> Any:
    if paddle.is_tensor(x=obj):
        placeholder = _TensorPlaceholder(index=len(tensors))
        tensors.append(obj)
        return placeholder
    elif isinstance(obj, dict):
        return {k: _split_tensors_from_obj(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_split_tensors_from_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_split_tensors_from_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_split_tensors_from_obj(v, tensors) for v in obj}
    else:
        return obj


def _put_tensors_in_obj(obj: Any, tensors: List[paddle.Tensor]) -> Any:
    if isinstance(obj, _TensorPlaceholder):
        return tensors[obj.index]
    elif isinstance(obj, dict):
        return {k: _put_tensors_in_obj(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_put_tensors_in_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_put_tensors_in_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_put_tensors_in_obj(v, tensors) for v in obj}
    else:
        return obj
