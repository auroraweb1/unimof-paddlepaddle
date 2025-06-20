import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
import ast
import collections
import logging
import os
import re
import shutil
import traceback
from typing import Any, Dict, Optional

import paddle
from paddle_utils import *

logger = logging.getLogger(__name__)


def ckp_copy_fun(src, checkpoints, end_of_epoch, args):
    has_copy = False
    can_delete = args.tmp_save_dir != args.save_dir
    for cp in checkpoints:
        try:
            if src != cp:
                logger.info("copy {} to {}".format(src, cp))
                has_copy = True
                shutil.copyfile(src, cp)
        except:
            logger.info("copy failed, please copy it manaully")
    try:
        if can_delete and has_copy and os.path.lexists(src):
            logger.info("removing temp file {} ...".format(src))
            os.remove(src)

        def remove_ckps(root_path):
            if not end_of_epoch and args.keep_interval_updates > 0:
                checkpoints = checkpoint_paths(
                    root_path, pattern="checkpoint_\\d+_(\\d+)\\.pt"
                )
                for old_chk in checkpoints[args.keep_interval_updates :]:
                    if os.path.lexists(old_chk):
                        os.remove(old_chk)
                        logger.info("removed {}".format(old_chk))
            if args.keep_last_epochs >= 0:
                checkpoints = checkpoint_paths(
                    root_path, pattern="checkpoint(\\d+)\\.pt"
                )
                for old_chk in checkpoints[args.keep_last_epochs :]:
                    if os.path.lexists(old_chk):
                        os.remove(old_chk)
                        logger.info("removed {}".format(old_chk))
            if args.keep_best_checkpoints > 0:
                checkpoints = checkpoint_paths(
                    root_path,
                    pattern="checkpoint\\.best_{}_(\\d+\\.?\\d*)\\.pt".format(
                        args.best_checkpoint_metric
                    ),
                )
                if not args.maximize_best_checkpoint_metric:
                    checkpoints = checkpoints[::-1]
                for old_chk in checkpoints[args.keep_best_checkpoints :]:
                    if os.path.lexists(old_chk):
                        os.remove(old_chk)
                        logger.info("removed {}".format(old_chk))

        remove_ckps(args.save_dir)
    except:
        logger.info("remove old ckps error")
    logger.info("finished async ckp saving.")


def save_checkpoint(args, trainer, epoch_itr, val_loss, ckp_copy_thread, do_save=True):
    from unicore import meters

    if trainer.data_parallel_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if args.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)
    if args.no_save or not do_save:
        return
    if not trainer.should_save_checkpoint_on_current_rank:
        return
    write_timer = meters.StopwatchMeter()
    """Not Support auto convert *.start, please judge whether it is Pytorch API and convert by yourself"""
    write_timer.start()
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()
    logger.info(f"Preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(a, b):
        return a >= b if args.maximize_best_checkpoint_metric else a <= b

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
        end_of_epoch
        and not args.no_epoch_checkpoints
        and epoch % args.save_interval == 0
    )
    (checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)]) = (
        not end_of_epoch
        and args.save_interval_updates > 0
        and updates % args.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )
    if val_loss is not None and args.keep_best_checkpoints > 0:
        checkpoint_conds[
            "checkpoint.best_{}_{:.2f}.pt".format(args.best_checkpoint_metric, val_loss)
        ] = not hasattr(save_checkpoint, "best") or is_better(
            val_loss, save_checkpoint.best
        )
    checkpoint_conds[
        "checkpoint_last{}.pt".format(suffix)
    ] = not args.no_last_checkpoints
    extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})
    checkpoints = [
        os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    tmp_checkpoints = [
        os.path.join(args.tmp_save_dir, fn)
        for fn, cond in checkpoint_conds.items()
        if cond
    ]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(tmp_checkpoints[0], extra_state)
        if ckp_copy_thread is not None:
            ckp_copy_thread.apply_async(
                ckp_copy_fun, (tmp_checkpoints[0], checkpoints, end_of_epoch, args)
            )
        """Not Support auto convert *.stop, please judge whether it is Pytorch API and convert by yourself"""
        write_timer.stop()
        logger.info(
            "Saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                tmp_checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )


def load_checkpoint(args, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """
    reset_optimizer = args.reset_optimizer
    reset_lr_scheduler = args.reset_lr_scheduler
    optimizer_overrides = ast.literal_eval(args.optimizer_overrides)
    reset_meters = args.reset_meters
    reset_dataloader = args.reset_dataloader
    if args.finetune_from_model is not None and (
        reset_optimizer or reset_lr_scheduler or reset_meters or reset_dataloader
    ):
        raise ValueError(
            "--finetune-from-model can not be set together with either --reset-optimizer or reset_lr_scheduler or reset_meters or reset_dataloader"
        )
    suffix = trainer.checkpoint_suffix
    if args.restore_file == "checkpoint_last.pt":
        checkpoint_path = os.path.join(
            args.save_dir, "checkpoint_last{}.pt".format(suffix)
        )
        first_launch = not os.path.exists(checkpoint_path)
        if args.finetune_from_model is not None and first_launch:
            if os.path.exists(args.finetune_from_model):
                checkpoint_path = args.finetune_from_model
                reset_optimizer = True
                reset_lr_scheduler = True
                reset_meters = True
                reset_dataloader = True
                logger.info(
                    f"loading pretrained model from {checkpoint_path}: optimizer, lr scheduler, meters, dataloader will be reset"
                )
            else:
                raise ValueError(
                    f"--funetune-from-model {args.finetune_from_model} does not exist"
                )
    elif suffix is not None:
        checkpoint_path = args.restore_file.replace(".pt", suffix + ".pt")
    else:
        checkpoint_path = args.restore_file
    if args.restore_file != "checkpoint_last.pt" and args.finetune_from_model:
        raise ValueError(
            "--finetune-from-model and --restore-file (non-default value) can not be specified together: "
            + str(args)
        )
    extra_state, epoch_itr = trainer.load_checkpoint(
        checkpoint_path,
        reset_optimizer,
        reset_lr_scheduler,
        reset_dataloader,
        optimizer_overrides,
        reset_meters=reset_meters,
        **passthrough_args,
    )
    if (
        extra_state is not None
        and "best" in extra_state
        and not reset_optimizer
        and not reset_meters
    ):
        save_checkpoint.best = extra_state["best"]
    return extra_state, epoch_itr


def load_checkpoint_to_cpu(path, arg_overrides=None, load_on_all_ranks=True):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).
    There's currently no support for > 1 but < all processes loading the
    checkpoint on each node.
    """
    local_path = path
    state = paddle.load(path=local_path)
    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
    return state


def checkpoint_paths(path, pattern="checkpoint(\\d+)\\.pt"):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)
    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(obj, filename):
    with open(filename + ".tmp", "wb") as f:
        _torch_persistent_save(obj, f)
    """Not Support auto convert *.rename, please judge whether it is Pytorch API and convert by yourself"""
    os.rename(filename + ".tmp", filename)


def _torch_persistent_save(obj, f):
    if isinstance(f, str):
        with open(f, "wb") as h:
            torch_persistent_save(obj, h)
        return
    for i in range(3):
        try:
            return paddle.save(obj=obj, path=f)
        except Exception:
            if i == 2:
                logger.error(traceback.format_exc())


def verify_checkpoint_directory(save_dir: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.join(save_dir, "dummy")
    try:
        with open(temp_file_path, "w"):
            pass
    except OSError as e:
        logger.warning(
            "Unable to access checkpoint save directory: {}".format(save_dir)
        )
        raise e
    else:
        os.remove(temp_file_path)
