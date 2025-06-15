import paddle

from . import UnicoreOptimizer, register_optimizer


@register_optimizer("sgd")
class SGD(UnicoreOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = paddle.optimizer.SGD(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument(
            "--momentum", default=0.0, type=float, metavar="M", help="momentum factor"
        )
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
        return {
            "lr": self.args.lr[0],
            "momentum": self.args.momentum,
            "weight_decay": self.args.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return True
