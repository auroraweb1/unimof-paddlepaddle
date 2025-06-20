from unicore.optim.lr_scheduler import (UnicoreLRScheduler,
                                        register_lr_scheduler)


@register_lr_scheduler("pass_through")
class PassThroughScheduleSchedule(UnicoreLRScheduler):
    """Delegate lr scheduling to the optimizer."""

    def __init__(self, args, optimizer, total_train_steps):
        super().__init__(args, optimizer, total_train_steps)
        assert (
            hasattr(optimizer, "lr_scheduler") and optimizer.lr_scheduler is not None
        ), "Pass-through schedule can only be used with optimizers with their own schedulers"

    def state_dict(self):
        return self.optimizer.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.lr_scheduler.load_state_dict(state_dict)

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        return self.optimizer.lr_scheduler.step_begin_epoch(epoch)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.lr_scheduler.step_update(num_updates)
