import sys
import os
import logging
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paddle
from paddle_utils import *
from unicore import checkpoint_utils, distributed_utils, options, tasks, utils
from unicore.logging import progress_bar

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimof.inference")


def main(args):
    """
    主函数，执行模型验证流程。
    Args:
        args (argparse.Namespace): 命令行参数对象
    Returns:
        None
    Raises:
        AssertionError: 如果未指定批处理大小
        Exception: 如果找不到指定的数据集
    """
    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"
    use_fp16 = args.fp16
    use_cuda = paddle.device.cuda.device_count() >= 1 and not args.cpu
    if use_cuda:
        paddle.device.set_device(device=device2str(args.device_id))
    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    
    # Fix weight shape mismatch by transposing specific layers
    for key in list(state.keys()):
        if any(k in key for k in ['fc1.weight', 'fc2.weight', 'in_proj.weight', 'linear1.weight', 'linear2.weight', 'out_proj.weight']):
            state[key] = state[key].transpose([1, 0])
    
    model.set_state_dict(state_dict=state)
    if use_fp16:
        model.astype(dtype="float16")
    if use_cuda:
        model.cuda()
    logger.info(args)
    loss = task.build_loss(args)
    loss.eval()
    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        fname = args.path.split("/")[-2]
        save_path = os.path.join(args.results_path, fname + "_" + subset + ".out.pkl")
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=0,  # Set to 0 to avoid signal handling in non-main thread
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format="tqdm" if not args.no_progress_bar else "simple",
        )
        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            _, _, log_output = task.valid_step(sample, model, loss, test=True)
            progress.log({}, step=i)
            log_outputs.append(log_output)
        # Convert tensors to numpy arrays before pickling
        processed_log_outputs = []
        for log in log_outputs:
            processed_log = {}
            for k, v in log.items():
                if hasattr(v, 'numpy'):
                    processed_log[k] = v.numpy()
                else:
                    processed_log[k] = v
            processed_log_outputs.append(processed_log)
        pickle.dump(processed_log_outputs, open(save_path, "wb"))
        logger.info("Done inference! ")
    return None


def cli_main():
    """
    命令行主函数。
    """
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
