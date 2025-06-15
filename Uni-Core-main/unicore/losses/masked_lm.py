import math

import paddle
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("masked_lm")
class MaskedLMLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()

    def forward(self, model, sample, reduce=True):
        masked_tokens = sample["target"].not_equal(y=paddle.to_tensor(self.padding_idx))
        sample_size = masked_tokens.astype(dtype="int32").sum()
        masked_tokens = paddle.where(
            condition=masked_tokens.astype("bool").any(),
            x=masked_tokens,
            y=masked_tokens.new([True]),
        )
        logits = model(**sample["net_input"], masked_tokens=masked_tokens)
        target = sample["target"]
        if masked_tokens is not None:
            target = target[masked_tokens]
        loss = paddle.nn.functional.nll_loss(
            input=paddle.nn.functional.log_softmax(x=logits, axis=-1, dtype="float32"),
            label=target,
            ignore_index=self.padding_idx,
            reduction="sum",
        )
        logging_output = {
            "loss": loss.data,
            "bsz": sample["target"].shape[0],
            "sample_size": sample_size,
            "seq_len": sample["target"].shape[1] * sample["target"].shape[0],
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
