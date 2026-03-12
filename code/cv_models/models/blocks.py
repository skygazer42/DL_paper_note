
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

