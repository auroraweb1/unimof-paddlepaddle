from typing import List

import paddle
from unicore.optim.lr_scheduler import (UnicoreLRScheduler,
                                        register_lr_scheduler)


@register_lr_scheduler("reduce_lr_on_plateau")
class ReduceLROnPlateauLRSchedule(UnicoreLRScheduler):
    """
    Decay the LR by a factor every time the validation loss plateaus.
    Also comes with optional warmup phase, where we linearly increase
    the learning rate from some initial learning rate
    (``--warmup-init-lr``) until the configured learning rate
    (``--lr``). Thereafter the lr is adjusted according to original
    reduce_on_plateau scheme.

    During warmup::

      lrs = torch.linspace(
          args.warmup_init_lr, args.lr, args.warmup_updates
      )
      lr = lrs[update_num]
    """

    def __init__(self, args, optimizer, total_train_steps):
        super().__init__(args, optimizer, total_train_steps)
        if len(args.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with reduce_lr_on_plateau. Consider --lr-scheduler=fixed instead."
            )
        tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(
            patience=args.lr_patience,
            factor=args.lr_shrink,
            mode="max" if args.maximize_best_checkpoint_metric else "min",
            threshold=args.lr_threshold,
            learning_rate=self.optimizer.optimizer.get_lr(),
        )
        self.optimizer.optimizer.set_lr_scheduler(tmp_lr)
        self.lr_scheduler = tmp_lr
        warmup_end_lr = args.lr[0]
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr
        if args.warmup_updates > 0:
            self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates
        self.warmup_end = True if args.warmup_updates <= 0 else False
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument(
            "--lr-shrink",
            default=0.1,
            type=float,
            metavar="LS",
            help="shrink factor for annealing, lr_new = (lr * lr_shrink)",
        )
        parser.add_argument(
            "--lr-threshold",
            default=0.0001,
            type=float,
            metavar="LT",
            help="Threshold for measuring the new optimum,                             to only focus on significant changes",
        )
        parser.add_argument(
            "--warmup-updates",
            default=0,
            type=int,
            metavar="N",
            help="warmup the learning rate linearly for the first N updates",
        )
        parser.add_argument(
            "--warmup-init-lr",
            default=-1,
            type=float,
            metavar="LR",
            help="initial learning rate during warmup phase; default is args.lr",
        )

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {
            "best": self.lr_scheduler.best,
            "last_epoch": self.lr_scheduler.last_epoch,
        }

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.lr_scheduler.best = state_dict["best"]
        if "last_epoch" in state_dict:
            self.lr_scheduler.last_epoch = state_dict["last_epoch"]

    def step(self, epoch, val_loss=None):
        """
        Update the learning rate at the end of the given epoch if warmup
        finishes otherwise no update of lr on epoch boundaries
        """
        if val_loss is not None and self.warmup_end is True:
            self.lr_scheduler.step(val_loss)
        else:
            self.lr_scheduler.last_epoch = epoch
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """
        Update the learning rate after each update."""
        if self.args.warmup_updates > 0:
            if num_updates <= self.args.warmup_updates:
                self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
                self.optimizer.set_lr(self.lr)
            elif self.warmup_end is False:
                self.warmup_end = True
        return self.optimizer.get_lr()
