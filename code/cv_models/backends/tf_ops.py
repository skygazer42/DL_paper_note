from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TensorFlowOps:
    dtype: Any = None

    def __post_init__(self):
        import tensorflow as tf

        self.dtype = tf.float32

    def asarray(self, x: Any):
        import tensorflow as tf

        if isinstance(x, (tf.Tensor, tf.Variable)):
            if x.dtype.is_floating:
                return tf.cast(x, self.dtype)
            return x
        if hasattr(x, "dtype") and getattr(x.dtype, "kind", None) in {"i", "u"}:
            return tf.convert_to_tensor(x, dtype=tf.int64)
        if hasattr(x, "dtype") and getattr(x.dtype, "kind", None) == "b":
            return tf.convert_to_tensor(x, dtype=tf.bool)
        return tf.convert_to_tensor(x, dtype=self.dtype)

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
        import tensorflow as tf

        x = self.asarray(x)  # NHWC
        w = self.asarray(w)  # (kH,kW,Cin/groups,Cout)
        if b is not None:
            b = self.asarray(b)

        if padding > 0:
            x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        strides = [1, stride, stride, 1]
        dilations = [1, dilation, dilation, 1]

        if groups == 1:
            y = tf.nn.conv2d(x, w, strides=strides, padding="VALID", dilations=dilations)
        else:
            # Split by input channels and output channels.
            x_splits = tf.split(x, num_or_size_splits=groups, axis=3)
            w_splits = tf.split(w, num_or_size_splits=groups, axis=3)
            ys = [
                tf.nn.conv2d(xg, wg, strides=strides, padding="VALID", dilations=dilations)
                for xg, wg in zip(x_splits, w_splits)
            ]
            y = tf.concat(ys, axis=3)
        if b is not None:
            y = y + b
        return y

    def linear(self, x, w, b=None):
        import tensorflow as tf

        x = self.asarray(x)
        w = self.asarray(w)
        y = tf.linalg.matmul(x, w)
        if b is not None:
            y = y + self.asarray(b)
        return y

    def relu(self, x):
        import tensorflow as tf

        x = self.asarray(x)
        return tf.nn.relu(x)

    def sigmoid(self, x):
        import tensorflow as tf

        x = self.asarray(x)
        return tf.nn.sigmoid(x)

    def gelu(self, x):
        import tensorflow as tf

        x = self.asarray(x)
        return tf.nn.gelu(x)

    def softmax(self, x, axis: int = -1):
        import tensorflow as tf

        x = self.asarray(x)
        return tf.nn.softmax(x, axis=axis)

    def max_pool2d(self, x, *, kernel: int = 2, stride: int | None = None, padding: int = 0):
        import tensorflow as tf

        if stride is None:
            stride = kernel
        x = self.asarray(x)
        if padding > 0:
            x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        return tf.nn.max_pool2d(x, ksize=kernel, strides=stride, padding="VALID")

    def avg_pool2d(self, x, *, kernel: int = 2, stride: int | None = None, padding: int = 0):
        import tensorflow as tf

        if stride is None:
            stride = kernel
        x = self.asarray(x)
        if padding > 0:
            x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        return tf.nn.avg_pool2d(x, ksize=kernel, strides=stride, padding="VALID")

    def global_avg_pool2d(self, x):
        import tensorflow as tf

        x = self.asarray(x)
        return tf.reduce_mean(x, axis=[1, 2])

    def upsample2d_nearest(self, x, *, scale: int = 2):
        import tensorflow as tf

        x = self.asarray(x)
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        return tf.image.resize(x, (h * scale, w * scale), method="nearest")

    def concat(self, xs: list, axis: int):
        import tensorflow as tf

        xs = [self.asarray(x) for x in xs]
        return tf.concat(xs, axis=axis)

    def reshape(self, x, shape: tuple[int, ...]):
        import tensorflow as tf

        x = self.asarray(x)
        return tf.reshape(x, shape)

    def transpose(self, x, axes: tuple[int, ...]):
        import tensorflow as tf

        x = self.asarray(x)
        return tf.transpose(x, perm=axes)

    def layer_norm(self, x, *, eps: float = 1e-5):
        import tensorflow as tf

        x = self.asarray(x)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) / tf.sqrt(var + eps)

    def matmul(self, a, b):
        import tensorflow as tf

        return tf.linalg.matmul(self.asarray(a), self.asarray(b))

    def gather(self, params, indices, *, axis: int = 0):
        import tensorflow as tf

        params = self.asarray(params)
        indices = self.asarray(indices)
        return tf.gather(params, indices, axis=axis)

    def reduce_sum(self, x, *, axis=None, keepdims: bool = False):
        import tensorflow as tf

        x = self.asarray(x)
        return tf.reduce_sum(x, axis=axis, keepdims=keepdims)

    def reduce_mean(self, x, *, axis=None, keepdims: bool = False):
        import tensorflow as tf

        x = self.asarray(x)
        return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
