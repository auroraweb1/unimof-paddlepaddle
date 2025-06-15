import paddle

from . import UnicoreOptimizer, register_optimizer


@register_optimizer("adadelta")
class Adadelta(UnicoreOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = paddle.optimizer.Adadelta(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument(
            "--adadelta-rho",
            type=float,
            default=0.9,
            metavar="RHO",
            help="coefficient used for computing a running average of squared gradients",
        )
        parser.add_argument(
            "--adadelta-eps",
            type=float,
            default=1e-06,
            metavar="EPS",
            help="term added to the denominator to improve numerical stability",
        )
        parser.add_argument(
            "--weight-decay",
            "--wd",
            default=0.0,
            type=float,
            metavar="WD",
            help="weight decay",
        )
        parser.add_argument(
            "--anneal-eps", action="store_true", help="flag to anneal eps"
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
            "rho": self.args.adadelta_rho,
            "eps": self.args.adadelta_eps,
            "weight_decay": self.args.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return True
