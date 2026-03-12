
# NOTE: Per request, this file is fully self-contained (no internal imports).
# TensorFlow is imported lazily when the model is instantiated.

import zlib
from typing import Any

NUM_CLASSES = 10
NUM_SEG_CLASSES = 3

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

from dataclasses import dataclass
from typing import Any

import numpy as np


def _fan_in_conv(k_h: int, k_w: int, in_ch: int, groups: int) -> int:
    return max(1, (k_h * k_w * in_ch) // max(1, groups))


@dataclass
class ParamBuilder:
    backend: Any
    seed: int

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def _randn(self, shape: tuple[int, ...], *, scale: float) -> Any:
        arr = (self.rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32, copy=False)
        return self.backend.asarray(arr)

    def zeros(self, shape: tuple[int, ...]) -> Any:
        arr = np.zeros(shape, dtype=np.float32)
        return self.backend.asarray(arr)

    def conv2d(self, in_ch: int, out_ch: int, *, k: int = 3, groups: int = 1) -> tuple[Any, Any]:
        fan_in = _fan_in_conv(k, k, in_ch, groups)
        scale = 1.0 / np.sqrt(float(fan_in))
        w = self._randn((k, k, in_ch // groups, out_ch), scale=scale)
        b = self.zeros((out_ch,))
        return w, b

    def linear(self, in_features: int, out_features: int) -> tuple[Any, Any]:
        fan_in = max(1, in_features)
        scale = 1.0 / np.sqrt(float(fan_in))
        w = self._randn((in_features, out_features), scale=scale)
        b = self.zeros((out_features,))
        return w, b

    def embedding(self, num_embeddings: int, embedding_dim: int) -> Any:
        scale = 1.0 / np.sqrt(float(max(1, embedding_dim)))
        return self._randn((num_embeddings, embedding_dim), scale=scale)

from typing import Any


def _n_hw_c(x) -> tuple[int, int, int, int]:
    n, h, w, c = x.shape
    return int(n), int(h), int(w), int(c)


def conv_relu(ops: Any, x, w, b, *, stride: int = 1, padding: int | None = None, groups: int = 1, dilation: int = 1):
    k = int(w.shape[0])
    if padding is None:
        padding = (k // 2) * dilation
    y = ops.conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return ops.relu(y)


def depthwise_separable_conv(ops: Any, x, w_dw, b_dw, w_pw, b_pw, *, stride: int = 1):
    # Depthwise: groups = in_channels
    _, _, _, c_in = _n_hw_c(x)
    y = ops.conv2d(x, w_dw, b_dw, stride=stride, padding=int(w_dw.shape[0]) // 2, groups=c_in)
    y = ops.relu(y)
    y = ops.conv2d(y, w_pw, b_pw, stride=1, padding=0)
    return ops.relu(y)


def se_block(ops: Any, x, w1, b1, w2, b2):
    n, _, _, c = _n_hw_c(x)
    s = ops.global_avg_pool2d(x)  # (N, C)
    s = ops.relu(ops.linear(s, w1, b1))
    s = ops.sigmoid(ops.linear(s, w2, b2))
    s = ops.reshape(s, (n, 1, 1, c))
    return x * s


def residual_block(ops: Any, x, w1, b1, w2, b2, *, stride: int = 1, proj: tuple[Any, Any] | None = None):
    y = conv_relu(ops, x, w1, b1, stride=stride)
    y = ops.conv2d(y, w2, b2, stride=1, padding=int(w2.shape[0]) // 2)
    skip = x
    if proj is not None:
        w_p, b_p = proj
        skip = ops.conv2d(x, w_p, b_p, stride=stride, padding=0)
    return ops.relu(y + skip)


def inception_block(ops: Any, x, b1, b3, b5, bpool):
    # Each branch is a (w,b) tuple.
    w1, bias1 = b1
    w3, bias3 = b3
    w5, bias5 = b5
    wp, bp = bpool

    p1 = conv_relu(ops, x, w1, bias1, padding=0)
    p3 = conv_relu(ops, x, w3, bias3)
    p5 = conv_relu(ops, x, w5, bias5, padding=2)
    pool = ops.max_pool2d(x, kernel=3, stride=1, padding=1)
    pp = conv_relu(ops, pool, wp, bp, padding=0)
    return ops.concat([p1, p3, p5, pp], axis=-1)


def channel_shuffle_nhwc(ops: Any, x, *, groups: int):
    n, h, w, c = _n_hw_c(x)
    if c % groups != 0:
        return x
    x = ops.reshape(x, (n, h, w, groups, c // groups))
    x = ops.transpose(x, (0, 1, 2, 4, 3))
    return ops.reshape(x, (n, h, w, c))


def patch_embed(ops: Any, image, w, b, *, patch: int, embed_dim: int):
    # Conv with stride=patch to produce (B, H/patch, W/patch, embed_dim)
    x = ops.conv2d(image, w, b, stride=patch, padding=0)
    bsz, h, w_, c = _n_hw_c(x)
    assert c == embed_dim
    tokens = ops.reshape(x, (bsz, h * w_, embed_dim))
    return tokens, (h, w_)


def mha(ops: Any, x, wq, bq, wk, bk, wv, bv, wo, bo, *, num_heads: int):
    # x: (B, T, D)
    bsz, t, d = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    if d % num_heads != 0:
        num_heads = 1
    head_dim = d // num_heads

    q = ops.linear(x, wq, bq)
    k = ops.linear(x, wk, bk)
    v = ops.linear(x, wv, bv)

    q = ops.reshape(q, (bsz, t, num_heads, head_dim))
    k = ops.reshape(k, (bsz, t, num_heads, head_dim))
    v = ops.reshape(v, (bsz, t, num_heads, head_dim))

    q = ops.transpose(q, (0, 2, 1, 3))  # (B, H, T, Dh)
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))

    k_t = ops.transpose(k, (0, 1, 3, 2))  # (B, H, Dh, T)
    scores = ops.matmul(q, k_t) * (head_dim ** -0.5)
    attn = ops.softmax(scores, axis=-1)
    ctx = ops.matmul(attn, v)  # (B, H, T, Dh)
    ctx = ops.transpose(ctx, (0, 2, 1, 3))  # (B, T, H, Dh)
    ctx = ops.reshape(ctx, (bsz, t, d))
    return ops.linear(ctx, wo, bo)


def transformer_encoder_block(ops: Any, x, params: dict[str, Any], *, prefix: str, num_heads: int):
    # Pre-norm
    x1 = ops.layer_norm(x)
    attn = mha(
        ops,
        x1,
        params[f"{prefix}.wq"],
        params[f"{prefix}.bq"],
        params[f"{prefix}.wk"],
        params[f"{prefix}.bk"],
        params[f"{prefix}.wv"],
        params[f"{prefix}.bv"],
        params[f"{prefix}.wo"],
        params[f"{prefix}.bo"],
        num_heads=num_heads,
    )
    x = x + attn

    x2 = ops.layer_norm(x)
    y = ops.gelu(ops.linear(x2, params[f"{prefix}.w1"], params[f"{prefix}.b1"]))
    y = ops.linear(y, params[f"{prefix}.w2"], params[f"{prefix}.b2"])
    return x + y

def _seed(model_id: str) -> int:
    # Stable across runs / machines.
    return int(zlib.adler32(model_id.encode("utf-8"))) & 0xFFFF_FFFF

def _classifier_head(ops: Any, x, w, b):
    pooled = ops.global_avg_pool2d(x)
    return ops.linear(pooled, w, b)

def _simple_backbone_params(pb: ParamBuilder, in_ch: int = 3, base: int = 16):
    # Returns a small 3-stage CNN backbone (for detection/segmentation)
    p: dict[str, Any] = {}
    p["c1.w"], p["c1.b"] = pb.conv2d(in_ch, base, k=3)
    p["c2.w"], p["c2.b"] = pb.conv2d(base, base * 2, k=3)
    p["c3.w"], p["c3.b"] = pb.conv2d(base * 2, base * 4, k=3)
    return p

def _simple_backbone_forward(ops: Any, x, p: dict[str, Any]):
    # Produces multi-scale features.
    x1 = conv_relu(ops, x, p["c1.w"], p["c1.b"], stride=2)  # 16x16
    x2 = conv_relu(ops, x1, p["c2.w"], p["c2.b"], stride=2)  # 8x8
    x3 = conv_relu(ops, x2, p["c3.w"], p["c3.b"], stride=2)  # 4x4
    return x1, x2, x3

def _mbconv_block(ops: Any, x, p: dict[str, Any], *, prefix: str, in_ch: int, out_ch: int, expansion: int, k: int, stride: int, se: bool):
    mid = in_ch * expansion
    # 1x1 expand
    y = conv_relu(ops, x, p[f"{prefix}.exp.w"], p[f"{prefix}.exp.b"], padding=0)
    # depthwise
    y = ops.conv2d(y, p[f"{prefix}.dw.w"], p[f"{prefix}.dw.b"], stride=stride, padding=k // 2, groups=mid)
    y = ops.relu(y)
    # se
    if se:
        y = se_block(ops, y, p[f"{prefix}.se.w1"], p[f"{prefix}.se.b1"], p[f"{prefix}.se.w2"], p[f"{prefix}.se.b2"])
    # project
    y = ops.conv2d(y, p[f"{prefix}.proj.w"], p[f"{prefix}.proj.b"], padding=0)
    if stride == 1 and in_ch == out_ch:
        return ops.relu(x + y)
    return ops.relu(y)

def _build_rcnn_family(ops: Any, model_id: str, *, with_rpn: bool, with_mask: bool, cascade: int):
    pb = ParamBuilder(ops, seed=_seed(model_id))
    p = _simple_backbone_params(pb, in_ch=3, base=16)

    # ROI head (global pooled feature from last stage)
    p["roi.fc1.w"], p["roi.fc1.b"] = pb.linear(64, 32)
    p["roi.cls.w"], p["roi.cls.b"] = pb.linear(32, NUM_CLASSES)
    p["roi.box.w"], p["roi.box.b"] = pb.linear(32, 4)

    if with_rpn:
        p["rpn.w"], p["rpn.b"] = pb.conv2d(64, 32, k=3)
        p["rpn.obj.w"], p["rpn.obj.b"] = pb.conv2d(32, 3, k=1)  # 3 anchors
        p["rpn.box.w"], p["rpn.box.b"] = pb.conv2d(32, 12, k=1)  # 3 * 4

    if with_mask:
        p["mask.w1"], p["mask.b1"] = pb.conv2d(64, 16, k=3)
        p["mask.w2"], p["mask.b2"] = pb.conv2d(16, NUM_SEG_CLASSES, k=1)

    # Cascade extra heads share the same hidden dim.
    for i in range(2, cascade + 1):
        p[f"roi{i}.box.w"], p[f"roi{i}.box.b"] = pb.linear(32, 4)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        _, _, feat = _simple_backbone_forward(ops, x, p)

        pooled = ops.global_avg_pool2d(feat)
        hidden = ops.relu(ops.linear(pooled, p["roi.fc1.w"], p["roi.fc1.b"]))
        out: dict[str, Any] = {
            "cls_logits": ops.linear(hidden, p["roi.cls.w"], p["roi.cls.b"]),
            "bbox_deltas": ops.linear(hidden, p["roi.box.w"], p["roi.box.b"]),
        }

        if with_rpn:
            rpn = conv_relu(ops, feat, p["rpn.w"], p["rpn.b"])
            out["rpn_objectness"] = ops.conv2d(rpn, p["rpn.obj.w"], p["rpn.obj.b"], padding=0)
            out["rpn_bbox"] = ops.conv2d(rpn, p["rpn.box.w"], p["rpn.box.b"], padding=0)

        if cascade > 1:
            for i in range(2, cascade + 1):
                out[f"bbox_deltas_{i}"] = ops.linear(hidden, p[f"roi{i}.box.w"], p[f"roi{i}.box.b"])

        if with_mask:
            m = conv_relu(ops, feat, p["mask.w1"], p["mask.b1"])
            m = ops.upsample2d_nearest(m, scale=2)
            out["mask_logits"] = ops.conv2d(m, p["mask.w2"], p["mask.b2"], padding=0)

        return out

    return forward

def _build_yolo_like(ops: Any, model_id: str, *, head_ch: int):
    pb = ParamBuilder(ops, seed=_seed(model_id))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["head.w"], p["head.b"] = pb.conv2d(64, head_ch, k=1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        _, _, feat = _simple_backbone_forward(ops, x, p)
        pred = ops.conv2d(feat, p["head.w"], p["head.b"], padding=0)
        return {"pred": pred}

    return forward

def _build_simple_segmentation(ops: Any, model_id: str, *, variant: str):
    pb = ParamBuilder(ops, seed=_seed(model_id))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["seg.w"], p["seg.b"] = pb.conv2d(64, NUM_SEG_CLASSES, k=1)

    # Decoder params
    p["dec.w1"], p["dec.b1"] = pb.conv2d(NUM_SEG_CLASSES, NUM_SEG_CLASSES, k=3)
    if variant in {"unet", "fusionnet", "deeplabv3_plus", "bisenet"}:
        p["skip.w"], p["skip.b"] = pb.conv2d(16, 8, k=1)
        p["fuse.w"], p["fuse.b"] = pb.conv2d(NUM_SEG_CLASSES + 8, NUM_SEG_CLASSES, k=3)

    # ASPP-ish for deeplab variants
    if variant in {"deeplabv3", "deeplabv3_plus"}:
        p["aspp1.w"], p["aspp1.b"] = pb.conv2d(64, 16, k=1)
        p["aspp2.w"], p["aspp2.b"] = pb.conv2d(64, 16, k=3)
        p["aspp3.w"], p["aspp3.b"] = pb.conv2d(64, 16, k=3)
        p["asppp.w"], p["asppp.b"] = pb.conv2d(48, 64, k=1)
    if variant in {"deeplabv1", "deeplabv2"}:
        p["atrous.w"], p["atrous.b"] = pb.conv2d(64, 64, k=3)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x1, _, feat = _simple_backbone_forward(ops, inputs["image"], p)

        if variant in {"deeplabv1", "deeplabv2"}:
            # Atrous conv effect via dilation
            feat = ops.relu(ops.conv2d(feat, p["atrous.w"], p["atrous.b"], padding=2, dilation=2))

        if variant in {"deeplabv3", "deeplabv3_plus"}:
            a1 = conv_relu(ops, feat, p["aspp1.w"], p["aspp1.b"], padding=0)
            a2 = conv_relu(ops, feat, p["aspp2.w"], p["aspp2.b"], dilation=2)
            a3 = conv_relu(ops, feat, p["aspp3.w"], p["aspp3.b"], dilation=4)
            feat = ops.concat([a1, a2, a3], axis=-1)
            feat = conv_relu(ops, feat, p["asppp.w"], p["asppp.b"], padding=0)

        seg = ops.conv2d(feat, p["seg.w"], p["seg.b"], padding=0)  # 4x4
        seg = ops.upsample2d_nearest(seg, scale=2)  # 8x8
        seg = conv_relu(ops, seg, p["dec.w1"], p["dec.b1"])
        seg = ops.upsample2d_nearest(seg, scale=4)  # 32x32

        if variant in {"unet", "fusionnet", "deeplabv3_plus", "bisenet"}:
            skip = conv_relu(ops, x1, p["skip.w"], p["skip.b"], padding=0)  # 16x16
            skip = ops.upsample2d_nearest(skip, scale=2)  # 32x32
            seg = ops.concat([seg, skip], axis=-1)
            seg = ops.conv2d(seg, p["fuse.w"], p["fuse.b"], padding=1)

        return {"seg_logits": seg}

    return forward

def _vit_like(ops: Any, model_id: str, *, patch: int, embed_dim: int, depth: int, heads: int):
    pb = ParamBuilder(ops, seed=_seed(model_id))
    p: dict[str, Any] = {}
    p["pe.w"], p["pe.b"] = pb.conv2d(3, embed_dim, k=patch)

    for i in range(depth):
        prefix = f"b{i}"
        p[f"{prefix}.wq"], p[f"{prefix}.bq"] = pb.linear(embed_dim, embed_dim)
        p[f"{prefix}.wk"], p[f"{prefix}.bk"] = pb.linear(embed_dim, embed_dim)
        p[f"{prefix}.wv"], p[f"{prefix}.bv"] = pb.linear(embed_dim, embed_dim)
        p[f"{prefix}.wo"], p[f"{prefix}.bo"] = pb.linear(embed_dim, embed_dim)
        p[f"{prefix}.w1"], p[f"{prefix}.b1"] = pb.linear(embed_dim, embed_dim * 2)
        p[f"{prefix}.w2"], p[f"{prefix}.b2"] = pb.linear(embed_dim * 2, embed_dim)

    p["head.w"], p["head.b"] = pb.linear(embed_dim, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        tokens, _ = patch_embed(ops, inputs["image"], p["pe.w"], p["pe.b"], patch=patch, embed_dim=embed_dim)
        for i in range(depth):
            tokens = transformer_encoder_block(ops, tokens, p, prefix=f"b{i}", num_heads=heads)
        pooled = ops.reduce_mean(tokens, axis=1)
        logits = ops.linear(pooled, p["head.w"], p["head.b"])
        return {"logits": logits}

    return forward


def _build_forward(ops):
    pb = ParamBuilder(ops, seed=_seed("senet"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)
    p["b1.w1"], p["b1.b1"] = pb.conv2d(16, 16, k=3)
    p["b1.w2"], p["b1.b2"] = pb.conv2d(16, 16, k=3)
    p["se.w1"], p["se.b1"] = pb.linear(16, 4)
    p["se.w2"], p["se.b2"] = pb.linear(4, 16)
    p["head.w"], p["head.b"] = pb.linear(16, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"])
        y = conv_relu(ops, x, p["b1.w1"], p["b1.b1"])
        y = ops.conv2d(y, p["b1.w2"], p["b1.b2"], padding=1)
        y = se_block(ops, y, p["se.w1"], p["se.b1"], p["se.w2"], p["se.b2"])
        x = ops.relu(x + y)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return forward


class SeNet:
    model_id = 'senet'

    def __init__(self, **kwargs: Any):
        self.ops = TensorFlowOps()
        self._forward = _build_forward(self.ops)

    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self._forward(inputs)

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self.forward(inputs)

MODEL_ID = 'senet'
MODEL_CLASS = SeNet
