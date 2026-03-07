from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TorchOps:
    dtype: Any = None

    def __post_init__(self):
        import torch

        self.dtype = torch.float32

    def asarray(self, x: Any):
        import torch

        if isinstance(x, torch.Tensor):
            if x.dtype.is_floating_point:
                return x.to(dtype=self.dtype)
            return x
        if hasattr(x, "dtype") and getattr(x.dtype, "kind", None) in {"i", "u"}:
            return torch.tensor(x, dtype=torch.long)
        if hasattr(x, "dtype") and getattr(x.dtype, "kind", None) == "b":
            return torch.tensor(x, dtype=torch.bool)
        return torch.tensor(x, dtype=self.dtype)

    def conv2d(
        self,
        x,
        w,
        b=None,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ):
        import torch.nn.functional as F

        x = self.asarray(x)  # NHWC
        w = self.asarray(w)  # (kH,kW,Cin/groups,Cout)
        if b is not None:
            b = self.asarray(b)

        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        w_oihw = w.permute(3, 2, 0, 1).contiguous()
        y = F.conv2d(x_nchw, w_oihw, bias=b, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y.permute(0, 2, 3, 1).contiguous()

    def linear(self, x, w, b=None):
        x = self.asarray(x)
        w = self.asarray(w)
        y = x.matmul(w)
        if b is not None:
            y = y + self.asarray(b)
        return y

    def relu(self, x):
        import torch

        x = self.asarray(x)
        return torch.relu(x)

    def sigmoid(self, x):
        import torch

        x = self.asarray(x)
        return torch.sigmoid(x)

    def gelu(self, x):
        import torch.nn.functional as F

        x = self.asarray(x)
        return F.gelu(x)

    def softmax(self, x, axis: int = -1):
        import torch

        x = self.asarray(x)
        return torch.softmax(x, dim=axis)

    def max_pool2d(self, x, *, kernel: int = 2, stride: int | None = None, padding: int = 0):
        import torch.nn.functional as F

        if stride is None:
            stride = kernel
        x = self.asarray(x)
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        y = F.max_pool2d(x_nchw, kernel_size=kernel, stride=stride, padding=padding)
        return y.permute(0, 2, 3, 1).contiguous()

    def avg_pool2d(self, x, *, kernel: int = 2, stride: int | None = None, padding: int = 0):
        import torch.nn.functional as F

        if stride is None:
            stride = kernel
        x = self.asarray(x)
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        y = F.avg_pool2d(x_nchw, kernel_size=kernel, stride=stride, padding=padding)
        return y.permute(0, 2, 3, 1).contiguous()

    def global_avg_pool2d(self, x):
        x = self.asarray(x)
        return x.mean(dim=(1, 2))

    def upsample2d_nearest(self, x, *, scale: int = 2):
        import torch

        x = self.asarray(x)
        x = torch.repeat_interleave(x, scale, dim=1)
        x = torch.repeat_interleave(x, scale, dim=2)
        return x

    def concat(self, xs: list, axis: int):
        import torch

        xs = [self.asarray(x) for x in xs]
        return torch.cat(xs, dim=axis)

    def reshape(self, x, shape: tuple[int, ...]):
        x = self.asarray(x)
        return x.reshape(shape)

    def transpose(self, x, axes: tuple[int, ...]):
        x = self.asarray(x)
        return x.permute(*axes).contiguous()

    def layer_norm(self, x, *, eps: float = 1e-5):
        import torch.nn.functional as F

        x = self.asarray(x)
        return F.layer_norm(x, normalized_shape=(x.shape[-1],), eps=eps)

    def matmul(self, a, b):
        a = self.asarray(a)
        b = self.asarray(b)
        return a.matmul(b)

    def gather(self, params, indices, *, axis: int = 0):
        import torch

        params = self.asarray(params)
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        if axis != 0:
            raise NotImplementedError("torch gather only implemented for axis=0 in this toy ops")
        return params.index_select(0, indices)

    def reduce_sum(self, x, *, axis=None, keepdims: bool = False):
        import torch

        x = self.asarray(x)
        if axis is None:
            return torch.sum(x)
        return torch.sum(x, dim=axis, keepdim=keepdims)

    def reduce_mean(self, x, *, axis=None, keepdims: bool = False):
        import torch

        x = self.asarray(x)
        if axis is None:
            return torch.mean(x)
        return torch.mean(x, dim=axis, keepdim=keepdims)
