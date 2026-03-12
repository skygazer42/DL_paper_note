
from dataclasses import dataclass
from typing import Any

import numpy as np


def _pad2d_nhwc(x: np.ndarray, padding: int) -> np.ndarray:
    if padding <= 0:
        return x
    return np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant")


def _as_strided_windows_2d(
    x: np.ndarray,
    *,
    kernel: tuple[int, int],
    stride: int,
    dilation: int,
) -> np.ndarray:
    # x is NHWC and already padded.
    x = np.ascontiguousarray(x)
    n, h, w, c = x.shape
    k_h, k_w = kernel

    out_h = (h - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (w - dilation * (k_w - 1) - 1) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid conv/pool output size: {(out_h, out_w)} from input {(h, w)}")

    s_n, s_h, s_w, s_c = x.strides
    shape = (n, out_h, out_w, k_h, k_w, c)
    strides = (
        s_n,
        s_h * stride,
        s_w * stride,
        s_h * dilation,
        s_w * dilation,
        s_c,
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


@dataclass
class NumpyOps:
    dtype: Any = np.float32

    def asarray(self, x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            if x.dtype.kind in {"i", "u", "b"}:
                return x
            return x.astype(self.dtype, copy=False)
        return np.asarray(x, dtype=self.dtype)

    def conv2d(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray | None = None,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> np.ndarray:
        """
        x: NHWC
        w: (kH, kW, C_in/groups, C_out)
        b: (C_out,)
        """
        x = self.asarray(x)
        w = self.asarray(w)
        if b is not None:
            b = self.asarray(b)

        n, h, w_in, c_in = x.shape
        k_h, k_w, c_in_g, c_out = w.shape
        if c_in % groups != 0:
            raise ValueError("input channels must be divisible by groups")
        if c_in_g != c_in // groups:
            raise ValueError("weight input channels must equal C_in/groups")
        if c_out % groups != 0:
            raise ValueError("output channels must be divisible by groups")

        x_pad = _pad2d_nhwc(x, padding)
        windows = _as_strided_windows_2d(x_pad, kernel=(k_h, k_w), stride=stride, dilation=dilation)

        c_out_g = c_out // groups
        outs = []
        for g in range(groups):
            xg = windows[..., g * c_in_g : (g + 1) * c_in_g]
            wg = w[..., g * c_out_g : (g + 1) * c_out_g]
            # tensordot over (kH, kW, C_in_g)
            yg = np.tensordot(xg, wg, axes=([3, 4, 5], [0, 1, 2]))
            outs.append(yg)
        y = np.concatenate(outs, axis=-1)
        if b is not None:
            y = y + b
        return y.astype(self.dtype, copy=False)

    def linear(self, x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
        x = self.asarray(x)
        w = self.asarray(w)
        y = x @ w
        if b is not None:
            y = y + self.asarray(b)
        return y.astype(self.dtype, copy=False)

    def relu(self, x: np.ndarray) -> np.ndarray:
        x = self.asarray(x)
        return np.maximum(x, 0.0, dtype=self.dtype)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = self.asarray(x)
        return (1.0 / (1.0 + np.exp(-x))).astype(self.dtype, copy=False)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        # tanh approximation
        x = self.asarray(x)
        return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))).astype(
            self.dtype, copy=False
        )

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = self.asarray(x)
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return (e / np.sum(e, axis=axis, keepdims=True)).astype(self.dtype, copy=False)

    def max_pool2d(self, x: np.ndarray, *, kernel: int = 2, stride: int | None = None, padding: int = 0):
        if stride is None:
            stride = kernel
        x = self.asarray(x)
        x_pad = _pad2d_nhwc(x, padding)
        windows = _as_strided_windows_2d(x_pad, kernel=(kernel, kernel), stride=stride, dilation=1)
        return windows.max(axis=(3, 4))

    def avg_pool2d(self, x: np.ndarray, *, kernel: int = 2, stride: int | None = None, padding: int = 0):
        if stride is None:
            stride = kernel
        x = self.asarray(x)
        x_pad = _pad2d_nhwc(x, padding)
        windows = _as_strided_windows_2d(x_pad, kernel=(kernel, kernel), stride=stride, dilation=1)
        return windows.mean(axis=(3, 4), dtype=self.dtype)

    def global_avg_pool2d(self, x: np.ndarray) -> np.ndarray:
        x = self.asarray(x)
        return x.mean(axis=(1, 2), dtype=self.dtype)

    def upsample2d_nearest(self, x: np.ndarray, *, scale: int = 2) -> np.ndarray:
        x = self.asarray(x)
        x = np.repeat(x, scale, axis=1)
        x = np.repeat(x, scale, axis=2)
        return x

    def concat(self, xs: list[np.ndarray], axis: int) -> np.ndarray:
        xs = [self.asarray(x) for x in xs]
        return np.concatenate(xs, axis=axis).astype(self.dtype, copy=False)

    def reshape(self, x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        x = self.asarray(x)
        return x.reshape(shape)

    def transpose(self, x: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        x = self.asarray(x)
        return x.transpose(axes)

    def layer_norm(self, x: np.ndarray, *, eps: float = 1e-5) -> np.ndarray:
        x = self.asarray(x)
        mean = x.mean(axis=-1, keepdims=True, dtype=self.dtype)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True, dtype=self.dtype)
        return ((x - mean) / np.sqrt(var + eps)).astype(self.dtype, copy=False)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (self.asarray(a) @ self.asarray(b)).astype(self.dtype, copy=False)

    def gather(self, params: np.ndarray, indices: np.ndarray, *, axis: int = 0) -> np.ndarray:
        params = self.asarray(params)
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices)
        if axis != 0:
            raise NotImplementedError("numpy gather only implemented for axis=0 in this toy ops")
        return params[indices]

    def reduce_sum(self, x: np.ndarray, *, axis=None, keepdims: bool = False) -> np.ndarray:
        x = self.asarray(x)
        return x.sum(axis=axis, keepdims=keepdims, dtype=self.dtype)

    def reduce_mean(self, x: np.ndarray, *, axis=None, keepdims: bool = False) -> np.ndarray:
        x = self.asarray(x)
        return x.mean(axis=axis, keepdims=keepdims, dtype=self.dtype)
