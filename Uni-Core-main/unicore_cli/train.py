import os

import paddle

"""
Train a new model on one or across multiple GPUs.
"""
import argparse
import logging
import math
import sys
import time
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from unicore import checkpoint_utils, options, tasks, utils
from unicore.data import iterators
from unicore.distributed import utils as distributed_utils
from unicore.logging import meters, metrics, progress_bar
from unicore.trainer import Trainer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unicore_cli.train")


def main(args) -> None:
    utils.import_user_module(args)
    utils.set_jit_fusion_options()
    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"
    metrics.reset()
    np.random.seed(args.seed)
    paddle.seed(seed=args.seed)
    if paddle.device.cuda.device_count() >= 1:
        paddle.seed(seed=args.seed)
    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)
        checkpoint_utils.verify_checkpoint_directory(args.tmp_save_dir)
        ckp_copy_thread = ThreadPool(processes=1)
    else:
        ckp_copy_thread = None
    logger.info(args)
    task = tasks.setup_task(args)
    assert args.loss, "Please specify loss to train a model"
    model = task.build_model(args)
    loss = task.build_loss(args)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("loss: {}".format(loss.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(getattr(p, "_orig_size", p).size for p in model.parameters()),
            sum(
                getattr(p, "_orig_size", p).size
                for p in model.parameters()
                if not p.stop_gradient
            ),
        )
    )
    trainer = Trainer(args, task, model, loss)
    logger.info("training on {} devices (GPUs)".format(args.distributed_world_size))
    logger.info("batch size per device = {}".format(args.batch_size))
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        args, trainer, disable_iterator_cache=False
    )
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    # Initialize training timer
    train_meter.reset()
    train_meter.start_time = time.time()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= args.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller than or equal to minimum learning rate (--stop-min-lr={args.stop_min_lr})"
            )
            break
        valid_losses, should_stop = train(
            args, trainer, task, epoch_itr, ckp_copy_thread
        )
        if should_stop:
            break
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            load_dataset=task.has_sharded_data("train"),
            disable_iterator_cache=False,
        )
    # Stop training timer
    train_meter.end_time = time.time()
    train_meter.sum = train_meter.end_time - train_meter.start_time
    if ckp_copy_thread is not None:
        ckp_copy_thread.close()
        ckp_copy_thread.join()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss: float) -> bool:
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    args, trainer: Trainer, task: tasks.UnicoreTask, epoch_itr, ckp_copy_thread
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=epoch_itr.next_epoch_idx > args.curriculum,
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=args.tensorboard_logdir
        if distributed_utils.is_master(args)
        else None,
        wandb_project=args.wandb_project if distributed_utils.is_master(args) else None,
        default_log_format="tqdm" if not args.no_progress_bar else "simple",
        args=args,
    )
    trainer.begin_epoch(epoch_itr.epoch)
    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"):
            # PaddlePaddle doesn't have direct profiler.record_function equivalent
            # Using context manager for metrics only
            log_output = trainer.train_step(samples)
        if log_output is not None:
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)
                metrics.reset_meters("train_inner")
        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch, ckp_copy_thread
        )
        if should_stop:
            break
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(
    args,
    trainer: Trainer,
    task: tasks.UnicoreTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    ckp_copy_thread,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to num_updates: {num_updates} >= max_update: {max_update}"
        )
    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if args.stop_time_hours > 0 and training_time_hours > args.stop_time_hours:
        should_stop = True
        logger.info(
            f"Stopping training due to cumulative_training_time: {training_time_hours} > stop_time_hours: {args.stop_time_hours} hour(s)"
        )
    do_save = (
        end_of_epoch
        and epoch_itr.epoch % args.save_interval == 0
        and not args.no_epoch_checkpoints
        or should_stop
        or args.save_interval_updates > 0
        and num_updates > 0
        and num_updates % args.save_interval_updates == 0
        and num_updates >= args.validate_after_updates
    )
    do_validate = (
        not end_of_epoch
        and do_save
        or end_of_epoch
        and epoch_itr.epoch % args.validate_interval == 0
        and not args.no_epoch_checkpoints
        or should_stop
        or args.validate_interval_updates > 0
        and num_updates > 0
        and num_updates % args.validate_interval_updates == 0
    ) and not args.disable_validation
    valid_losses = [None]
    if do_validate:
        with utils.validate_with_ema(trainer, ema=args.validate_with_ema):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
    should_stop |= should_stop_early(args, valid_losses[0])
    checkpoint_utils.save_checkpoint(
        args,
        trainer,
        epoch_itr,
        valid_losses[0],
        ckp_copy_thread,
        do_save=do_save or should_stop,
    )
    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    args, trainer: Trainer, task: tasks.UnicoreTask, epoch_itr, subsets: List[str]
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""
    seed = None
    if args.fixed_validation_seed is not None:
        seed = args.fixed_validation_seed
    with utils.paddle_seed(seed):
        trainer.begin_valid_epoch(epoch_itr.epoch)
        valid_losses = []
        for subset in subsets:
            logger.info('begin validation on "{}" subset'.format(subset))
            itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False
            )
            progress = progress_bar.progress_bar(
                itr,
                log_format=args.log_format,
                log_interval=args.log_interval,
                epoch=epoch_itr.epoch,
                prefix=f"valid on '{subset}' subset",
                tensorboard_logdir=args.tensorboard_logdir
                if distributed_utils.is_master(args)
                else None,
                default_log_format="tqdm" if not args.no_progress_bar else "simple",
            )
            with metrics.aggregate(new_root=True) as agg:
                logging_outputs = []
                for i, sample in enumerate(progress):
                    if args.max_valid_steps is not None and i > args.max_valid_steps:
                        break
                    inner_logging_outputs = trainer.valid_step(sample)
                    logging_outputs.extend(inner_logging_outputs)
                task.reduce_metrics(logging_outputs, trainer.get_loss(), subset)
            stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=trainer.get_num_updates())
            if args.best_checkpoint_metric in stats:
                valid_losses.append(stats[args.best_checkpoint_metric])
        return valid_losses


def get_valid_stats(args, trainer: Trainer, stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if (
        hasattr(checkpoint_utils.save_checkpoint, "best")
        and args.best_checkpoint_metric in stats
    ):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    try:
        if args.profile:
# PaddlePaddle profiling not implemented in this version
            distributed_utils.call_main(args, main)
        else:
            distributed_utils.call_main(args, main)
    finally:
        if paddle.distributed.is_initialized():
            paddle.distributed.barrier()
paddle.distributed.destroy_process_group()


if __name__ == "__main__":
    cli_main()
