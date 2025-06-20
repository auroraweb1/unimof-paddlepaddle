from typing import List

from unicore.optim.lr_scheduler import (UnicoreLRScheduler,
                                        register_lr_scheduler)


@register_lr_scheduler("fixed")
class FixedLRSchedule(UnicoreLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer, total_train_steps):
        super().__init__(args, optimizer, total_train_steps)
        self.lr = args.lr[0]
        if args.warmup_updates > 0:
            self.warmup_factor = 1.0 / args.warmup_updates
        else:
            self.warmup_factor = 1

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument(
            "--force-anneal",
            "--fa",
            type=int,
            metavar="N",
            help="force annealing at specified epoch",
        )
        parser.add_argument(
            "--lr-shrink",
            default=0.1,
            type=float,
            metavar="LS",
            help="shrink factor for annealing, lr_new = (lr * lr_shrink)",
        )
        parser.add_argument(
            "--warmup-updates",
            default=0,
            type=int,
            metavar="N",
            help="warmup the learning rate linearly for the first N updates",
        )

    def state_dict(self):
        return {"lr": self.lr}

    def load_state_dict(self, state_dict):
        if "lr" in state_dict:
            self.lr = state_dict["lr"]

    def get_next_lr(self, epoch):
        lrs = self.args.lr
        if self.args.force_anneal is None or epoch < self.args.force_anneal:
            next_lr = lrs[min(epoch - 1, len(lrs) - 1)]
        else:
            next_lr = lrs[-1] * self.args.lr_shrink ** (
                epoch + 1 - self.args.force_anneal
            )
        return next_lr

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.args.warmup_updates > 0 and num_updates < self.args.warmup_updates:
            self.warmup_factor = (num_updates + 1) / float(self.args.warmup_updates)
            self.optimizer.set_lr(self.warmup_factor * self.lr)
        else:
            self.optimizer.set_lr(self.lr)
        return self.optimizer.get_lr()
