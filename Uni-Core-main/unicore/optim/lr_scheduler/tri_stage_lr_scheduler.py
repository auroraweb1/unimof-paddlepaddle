import math
from typing import List

from unicore.optim.lr_scheduler import (UnicoreLRScheduler,
                                        register_lr_scheduler)


@register_lr_scheduler("tri_stage")
class TriStageLRSchedule(UnicoreLRScheduler):
    """Tristage learning rate schedulr

    Implement the learning rate scheduler in https://arxiv.org/pdf/1904.08779.pdf

    Similar to inverse_squre_root scheduler, but tri_stage learning rate employs
    three stages LR scheduling:

        - warmup stage, starting from `lr` * `init_lr_scale`, linearly
          increased to `lr` in `warmup_steps` iterations

        - hold stage, after `warmup_steps`, keep the LR as `lr` for `hold_steps`
          iterations

        - decay stage, after hold stage, decay LR exponetially to
          `lr` * `final_lr_scale` in `decay_steps`;
          after that LR is keep as `final_lr_scale` * `lr`

    During warmup::

      init_lr = args.init_lr_scale * args.lr
      lrs = torch.linspace(init_lr, args.lr, args.warmup_steps)
      lr = lrs[update_num]

    During hold::

      lr = args.lr

    During decay::

      decay_factor = - math.log(args.final_lr_scale) / args.decay_steps
      lr = args.lr * exp(- (update_num - warmup_steps - decay_steps) * decay_factor)

    After that::

      lr = args.lr * args.final_lr_scale
    """

    def __init__(self, args, optimizer, total_train_steps):
        super().__init__(args, optimizer, total_train_steps)
        if len(args.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with tri-stage lr. Consider --lr-scheduler=fixed instead."
            )
        self.peak_lr = args.lr[0]
        self.init_lr = args.init_lr_scale * args.lr[0]
        self.final_lr = args.final_lr_scale * args.lr[0]
        if args.phase_ratio is not None:
            assert args.max_update > 0
            assert sum(args.phase_ratio) == 1, "phase ratios must add up to 1"
            self.warmup_steps = int(args.max_update * args.phase_ratio[0])
            self.hold_steps = int(args.max_update * args.phase_ratio[1])
            self.decay_steps = int(args.max_update * args.phase_ratio[2])
        else:
            self.warmup_steps = args.warmup_steps
            self.hold_steps = args.hold_steps
            self.decay_steps = args.decay_steps
        assert (
            self.warmup_steps + self.hold_steps + self.decay_steps > 0
        ), "please specify steps or phase_ratio"
        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(args.final_lr_scale) / self.decay_steps
        self.lr = self.init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument(
            "--warmup-steps",
            default=4000,
            type=int,
            metavar="N",
            help="warmup the learning rate linearly for the first N updates",
        )
        parser.add_argument(
            "--hold-steps",
            default=20000,
            type=int,
            metavar="N",
            help="steps in hold stage.",
        )
        parser.add_argument(
            "--decay-steps",
            default=60000,
            type=int,
            metavar="N",
            help="steps in decay stages",
        )
        parser.add_argument(
            "--init-lr-scale",
            default=0.01,
            type=float,
            help="""
    initial learning rate scale during warmup phase; default is 0.01""",
        )
        parser.add_argument(
            "--final-lr-scale",
            default=0.01,
            type=float,
            help="final learning rate scale; default to 0.01",
        )

    def _decide_stage(self, update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            return 0, update_step
        offset = self.warmup_steps
        if update_step < offset + self.hold_steps:
            return 1, update_step - offset
        offset += self.hold_steps
        if update_step <= offset + self.decay_steps:
            return 2, update_step - offset
        offset += self.decay_steps
        return 3, update_step - offset

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        stage, steps_in_stage = self._decide_stage(num_updates)
        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")
        self.optimizer.set_lr(self.lr)
        return self.lr
