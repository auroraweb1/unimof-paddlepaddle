from collections.abc import Collection
from typing import List

from unicore.optim.lr_scheduler import (UnicoreLRScheduler,
                                        register_lr_scheduler)


@register_lr_scheduler("inverse_sqrt")
class InverseSquareRootSchedule(UnicoreLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer, total_train_steps):
        super().__init__(args, optimizer, total_train_steps)
        if isinstance(args.lr, Collection) and len(args.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with inverse_sqrt. Consider --lr-scheduler=fixed instead."
            )
        warmup_end_lr = args.lr[0] if isinstance(args.lr, Collection) else args.lr
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates
        self.decay_factor = warmup_end_lr * args.warmup_updates**0.5
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument(
            "--warmup-updates",
            default=4000,
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

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * num_updates**-0.5
        self.optimizer.set_lr(self.lr)
        return self.lr
