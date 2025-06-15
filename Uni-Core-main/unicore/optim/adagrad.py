import paddle

from . import UnicoreOptimizer, register_optimizer


@register_optimizer("adagrad")
class Adagrad(UnicoreOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = paddle.optimizer.Adagrad(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument(
            "--weight-decay",
            "--wd",
            default=0.0,
            type=float,
            metavar="WD",
            help="weight decay",
        )

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {"lr": self.args.lr[0], "weight_decay": self.args.weight_decay}

    @property
    def supports_flat_params(self):
        return False
