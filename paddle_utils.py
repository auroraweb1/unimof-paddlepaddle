
import paddle

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

class Embedding(paddle.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_idx = self._padding_idx

def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_max", _Tensor_max)

def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f'gpu:{type}'
    elif isinstance(type, str):
        if 'cuda' in type:
            type = type.replace('cuda', 'gpu')
        if 'cpu' in type:
            type = 'cpu'
        elif index is not None:
            type = f'{type}:{index}'
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = 'cpu'
    elif isinstance(type, paddle.CUDAPlace):
        type = f'gpu:{type.get_device_id()}'

    return type

def _Tensor_take(self, *args, **kwargs):
    if args:
        return paddle.take(self, *args)
    elif kwargs:
        return paddle.take(self, **kwargs)

setattr(paddle.Tensor, "take", _Tensor_take)

def device2int(device):
    if isinstance(device, str):
        device = device.replace('cuda', 'gpu')
        device = device.replace('gpu:', '')
    return int(device)

def _Tensor_view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', _Tensor_view)

def paddle_split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis]//num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)

def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm

def _Tensor_min(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_min", _Tensor_min)

def paddle_min(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(*args, **kwargs)
    elif len(args)==2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.minimum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                return ret
        else:
            ret = paddle.min(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret

def paddle_max(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(*args, **kwargs)
    elif len(args)==2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.maximum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                return ret
        else:
            ret = paddle.max(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret

def _Tensor_add(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        if alpha != 1:
            if not isinstance(y, paddle.Tensor):
                y = paddle.to_tensor(alpha * y)
            else:
                y = alpha * y
    else:
        if not isinstance(y, paddle.Tensor):
            y = paddle.to_tensor(y)
    return paddle.add(self, y)

setattr(paddle.Tensor, "add", _Tensor_add)

def _Tensor_reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])

setattr(paddle.Tensor, "reshape", _Tensor_reshape)
############################## 相关utils函数，如上 ##############################

