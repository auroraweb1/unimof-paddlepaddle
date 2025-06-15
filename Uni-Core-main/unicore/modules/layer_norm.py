import numbers

import paddle

try:
    import unicore_fused_layernorm
    import unicore_fused_layernorm_backward_gamma_beta

    HAS_LAYER_NORM = True
except:
    print("fused_layer_norm is not installed corrected")
    HAS_LAYER_NORM = False
if (
    not paddle.device.cuda.device_count() >= 1
    or paddle.device.cuda.get_device_capability()[0] < 7
):
    HAS_LAYER_NORM = False


class FusedLayerNormFastFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input = input.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        output, mean, invvar = unicore_fused_layernorm.forward(
            input, ctx.normalized_shape, weight, bias, ctx.eps
        )
        ctx.save_for_backward(input, weight, bias, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensor()
        grad_input = grad_weight = grad_bias = None
        grad_input = unicore_fused_layernorm.backward(
            grad_output.contiguous(),
            mean,
            invvar,
            input_,
            ctx.normalized_shape,
            weight_,
            bias_,
            ctx.eps,
        )
        grad_weight, grad_bias = unicore_fused_layernorm_backward_gamma_beta.backward(
            grad_output.contiguous(),
            mean,
            invvar,
            input_,
            ctx.normalized_shape,
            weight_,
            bias_,
            ctx.eps,
        )
        return grad_input, grad_weight, grad_bias, None, None


FUSED_LAYER_NORM_SUPPORT_DIM = set(
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


class LayerNorm(paddle.nn.Layer):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        assert elementwise_affine
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=normalized_shape)
        )
        self.bias = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=normalized_shape)
        )
        self.reset_parameters()

        def torch_layer_norm(input):
            return paddle.nn.functional.layer_norm(
                x=input,
                normalized_shape=self.normalized_shape,
                weight=self.weight.astype(input.dtype),
                bias=self.bias.astype(input.dtype),
                epsilon=self.eps,
            )

        def fused_layer_norm(input):
            if input.place.is_gpu_place():
                return FusedLayerNormFastFunction.apply(
                    input,
                    self.weight.astype(input.dtype),
                    self.bias.astype(input.dtype),
                    self.normalized_shape,
                    self.eps,
                )
            else:
                return paddle.nn.functional.layer_norm(
                    x=input,
                    normalized_shape=self.normalized_shape,
                    weight=self.weight.astype(input.dtype),
                    bias=self.bias.astype(input.dtype),
                    epsilon=self.eps,
                )

        self.func = (
            torch_layer_norm
            if not HAS_LAYER_NORM
            or normalized_shape[0] not in FUSED_LAYER_NORM_SUPPORT_DIM
            else fused_layer_norm
        )

    def reset_parameters(self):
        init_Constant = paddle.nn.initializer.Constant(value=1.0)
        init_Constant(self.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.bias)

    def forward(self, input):
        return self.func(input)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, elementwise_affine=True".format(
            **self.__dict__
        )
