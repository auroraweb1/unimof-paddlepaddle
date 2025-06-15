import numbers

import paddle

try:
    import unicore_fused_rmsnorm
    import unicore_fused_rmsnorm_backward_gamma

    HAS_RMS_NORM = True
except:
    print("fused_rms_norm is not installed corrected")
    HAS_RMS_NORM = False
if (
    not paddle.device.cuda.device_count() >= 1
    or paddle.device.cuda.get_device_capability()[0] < 7
):
    HAS_RMS_NORM = False


class FusedRMSNormFastFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input = input.contiguous()
        weight = weight.contiguous()
        output, invvar = unicore_fused_rmsnorm.forward(
            input, ctx.normalized_shape, weight, ctx.eps
        )
        ctx.save_for_backward(input, weight, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, invvar = ctx.saved_tensor()
        grad_input = grad_weight = None
        grad_input = unicore_fused_rmsnorm.backward(
            grad_output.contiguous(),
            invvar,
            input_,
            ctx.normalized_shape,
            weight_,
            ctx.eps,
        )
        grad_weight = unicore_fused_rmsnorm_backward_gamma.backward(
            grad_output.contiguous(),
            invvar,
            input_,
            ctx.normalized_shape,
            weight_,
            ctx.eps,
        )
        return grad_input, grad_weight, None, None


FUSED_RMS_NORM_SUPPORT_DIM = set(
    [
        64,
        128,
        192,
        256,
        320,
        384,
        512,
        640,
        768,
        1024,
        1280,
        1536,
        1792,
        2048,
        2560,
        5120,
    ]
)


class RMSNorm(paddle.nn.Layer):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        assert elementwise_affine
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=normalized_shape)
        )
        self.reset_parameters()

        def torch_rms_norm(input):
            return torch.nn.functional.rms_norm(
                input, self.normalized_shape, self.weight.astype(input.dtype), self.eps
            )

        def fused_rms_norm(input):
            if input.place.is_gpu_place():
                return FusedRMSNormFastFunction.apply(
                    input,
                    self.weight.astype(input.dtype),
                    self.normalized_shape,
                    self.eps,
                )
            else:
                return torch.nn.functional.rms_norm(
                    input,
                    self.normalized_shape,
                    self.weight.astype(input.dtype),
                    self.eps,
                )

        self.func = (
            torch_rms_norm
            if not HAS_RMS_NORM or normalized_shape[0] not in FUSED_RMS_NORM_SUPPORT_DIM
            else fused_rms_norm
        )

    def reset_parameters(self):
        init_Constant = paddle.nn.initializer.Constant(value=1.0)
        init_Constant(self.weight)

    def forward(self, input):
        return self.func(input)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, elementwise_affine=True".format(
            **self.__dict__
        )
