import sys

sys.path.append("E:\\beihang\\Uni-Mof-Main-PaddlePaddle")
import math

import paddle
from paddle_utils import *
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("cross_entropy")
class CrossEntropyLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        if isinstance(net_output, tuple):
            logits = net_output[0]
        else:
            logits = net_output
        loss = self.compute_loss(model, logits, sample, reduce=reduce)
        sample_size = sample["target"].shape[0]
        logging_output = {
            "loss": loss.data,
            "bsz": sample["target"].shape[0],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = paddle.nn.functional.log_softmax(
            x=net_output.astype(dtype="float32"), axis=-1
        )
        lprobs = lprobs.view(-1, lprobs.shape[-1])
        target = sample["target"].view(-1)
        loss = paddle.nn.functional.nll_loss(
            input=lprobs, label=target, reduction="sum" if reduce else "none"
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
