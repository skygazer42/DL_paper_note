
import zlib
from typing import Any, Callable

from .blocks import (
    channel_shuffle_nhwc,
    conv_relu,
    depthwise_separable_conv,
    inception_block,
    patch_embed,
    residual_block,
    se_block,
    transformer_encoder_block,
)
from .core import ForwardModel
from .initializers import ParamBuilder

NUM_CLASSES = 10
NUM_SEG_CLASSES = 3


def _seed(model_id: str) -> int:
    # 只依赖 model_id 生成稳定种子，保证同一个模型在不同机器上参数一致。
    return int(zlib.adler32(model_id.encode("utf-8"))) & 0xFFFF_FFFF


def _classifier_head(ops: Any, x, w, b):
    # 分类模型统一走 GAP + Linear，便于不同 backbone 复用同一套输出头。
    pooled = ops.global_avg_pool2d(x)
    return ops.linear(pooled, w, b)


def _simple_backbone_params(pb: ParamBuilder, in_ch: int = 3, base: int = 16):
    # 检测/分割类 toy 模型共享一个极小的 3-stage CNN 主干，重点演示接口而非还原大模型。
    p: dict[str, Any] = {}
    p["c1.w"], p["c1.b"] = pb.conv2d(in_ch, base, k=3)
    p["c2.w"], p["c2.b"] = pb.conv2d(base, base * 2, k=3)
    p["c3.w"], p["c3.b"] = pb.conv2d(base * 2, base * 4, k=3)
    return p


def _simple_backbone_forward(ops: Any, x, p: dict[str, Any]):
    # 返回 3 个尺度的特征图，供 FPN / head / decoder 一类结构复用。
    x1 = conv_relu(ops, x, p["c1.w"], p["c1.b"], stride=2)  # 16x16
    x2 = conv_relu(ops, x1, p["c2.w"], p["c2.b"], stride=2)  # 8x8
    x3 = conv_relu(ops, x2, p["c3.w"], p["c3.b"], stride=2)  # 4x4
    return x1, x2, x3


# -------------------------
# Classification models
# -------------------------


def build_bp(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("bp"))
    p: dict[str, Any] = {}
    p["fc1.w"], p["fc1.b"] = pb.linear(128, 64)
    p["fc2.w"], p["fc2.b"] = pb.linear(64, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["x"]
        x = ops.relu(ops.linear(x, p["fc1.w"], p["fc1.b"]))
        logits = ops.linear(x, p["fc2.w"], p["fc2.b"])
        return {"logits": logits}

    return ForwardModel("bp", ops, forward)


def build_zfnet(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("zfnet"))
    p: dict[str, Any] = {}
    p["c1.w"], p["c1.b"] = pb.conv2d(3, 16, k=7)
    p["c2.w"], p["c2.b"] = pb.conv2d(16, 32, k=5)
    p["c3.w"], p["c3.b"] = pb.conv2d(32, 64, k=3)
    p["head.w"], p["head.b"] = pb.linear(64, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        x = conv_relu(ops, x, p["c1.w"], p["c1.b"], stride=2, padding=3)
        x = ops.max_pool2d(x, kernel=2, stride=2)
        x = conv_relu(ops, x, p["c2.w"], p["c2.b"], padding=2)
        x = ops.max_pool2d(x, kernel=2, stride=2)
        x = conv_relu(ops, x, p["c3.w"], p["c3.b"])
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("zfnet", ops, forward)


def build_vgg(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("vgg"))
    p: dict[str, Any] = {}
    p["c11.w"], p["c11.b"] = pb.conv2d(3, 16, k=3)
    p["c12.w"], p["c12.b"] = pb.conv2d(16, 16, k=3)
    p["c21.w"], p["c21.b"] = pb.conv2d(16, 32, k=3)
    p["c22.w"], p["c22.b"] = pb.conv2d(32, 32, k=3)
    p["c31.w"], p["c31.b"] = pb.conv2d(32, 64, k=3)
    p["head.w"], p["head.b"] = pb.linear(64, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        x = conv_relu(ops, x, p["c11.w"], p["c11.b"])
        x = conv_relu(ops, x, p["c12.w"], p["c12.b"])
        x = ops.max_pool2d(x, kernel=2, stride=2)
        x = conv_relu(ops, x, p["c21.w"], p["c21.b"])
        x = conv_relu(ops, x, p["c22.w"], p["c22.b"])
        x = ops.max_pool2d(x, kernel=2, stride=2)
        x = conv_relu(ops, x, p["c31.w"], p["c31.b"])
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("vgg", ops, forward)


def build_nin(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("nin"))
    p: dict[str, Any] = {}
    p["c1.w"], p["c1.b"] = pb.conv2d(3, 16, k=5)
    p["n1.w"], p["n1.b"] = pb.conv2d(16, 32, k=1)
    p["n2.w"], p["n2.b"] = pb.conv2d(32, NUM_CLASSES, k=1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        x = conv_relu(ops, x, p["c1.w"], p["c1.b"], padding=2)
        x = conv_relu(ops, x, p["n1.w"], p["n1.b"], padding=0)
        x = ops.conv2d(x, p["n2.w"], p["n2.b"], padding=0)
        logits = ops.global_avg_pool2d(x)
        return {"logits": logits}

    return ForwardModel("nin", ops, forward)


def build_googlenet(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("googlenet"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)

    # 这里只保留 Inception 的多分支结构轮廓，规模故意压得很小。
    p["inc1.b1"] = pb.conv2d(16, 8, k=1)
    p["inc1.b3"] = pb.conv2d(16, 8, k=3)
    p["inc1.b5"] = pb.conv2d(16, 8, k=5)
    p["inc1.bp"] = pb.conv2d(16, 8, k=1)

    p["inc2.b1"] = pb.conv2d(32, 8, k=1)
    p["inc2.b3"] = pb.conv2d(32, 8, k=3)
    p["inc2.b5"] = pb.conv2d(32, 8, k=5)
    p["inc2.bp"] = pb.conv2d(32, 8, k=1)

    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        x = conv_relu(ops, x, p["stem.w"], p["stem.b"])
        x = inception_block(ops, x, p["inc1.b1"], p["inc1.b3"], p["inc1.b5"], p["inc1.bp"])
        x = ops.max_pool2d(x, kernel=2, stride=2)
        x = inception_block(ops, x, p["inc2.b1"], p["inc2.b3"], p["inc2.b5"], p["inc2.bp"])
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("googlenet", ops, forward)


def build_resnet(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("resnet"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)

    p["b1.w1"], p["b1.b1"] = pb.conv2d(16, 16, k=3)
    p["b1.w2"], p["b1.b2"] = pb.conv2d(16, 16, k=3)

    p["b2.w1"], p["b2.b1"] = pb.conv2d(16, 32, k=3)
    p["b2.w2"], p["b2.b2"] = pb.conv2d(32, 32, k=3)
    p["b2.wp"], p["b2.bp"] = pb.conv2d(16, 32, k=1)

    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        x = conv_relu(ops, x, p["stem.w"], p["stem.b"])
        x = residual_block(ops, x, p["b1.w1"], p["b1.b1"], p["b1.w2"], p["b1.b2"])
        x = residual_block(
            ops,
            x,
            p["b2.w1"],
            p["b2.b1"],
            p["b2.w2"],
            p["b2.b2"],
            stride=2,
            proj=(p["b2.wp"], p["b2.bp"]),
        )
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("resnet", ops, forward)


def build_senet(ops: Any) -> ForwardModel:
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

    return ForwardModel("senet", ops, forward)


def build_resnext(ops: Any) -> ForwardModel:
    # Tiny ResNeXt-style bottleneck with grouped conv
    pb = ParamBuilder(ops, seed=_seed("resnext"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)

    # Bottleneck: 1x1 -> 3x3 group -> 1x1
    p["b.w1"], p["b.b1"] = pb.conv2d(16, 16, k=1)
    p["b.w2"], p["b.b2"] = pb.conv2d(16, 16, k=3, groups=4)
    p["b.w3"], p["b.b3"] = pb.conv2d(16, 32, k=1)
    p["b.wp"], p["b.bp"] = pb.conv2d(16, 32, k=1)
    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"])
        y = conv_relu(ops, x, p["b.w1"], p["b.b1"], padding=0)
        y = conv_relu(ops, y, p["b.w2"], p["b.b2"], groups=4)
        y = ops.conv2d(y, p["b.w3"], p["b.b3"], padding=0)
        skip = ops.conv2d(x, p["b.wp"], p["b.bp"], padding=0)
        x = ops.relu(y + skip)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("resnext", ops, forward)


def build_densenet(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("densenet"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)
    p["d1.w"], p["d1.b"] = pb.conv2d(16, 8, k=3)
    p["d2.w"], p["d2.b"] = pb.conv2d(24, 8, k=3)  # concat(16+8)
    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)  # concat(16+8+8)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x0 = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"])
        x1 = conv_relu(ops, x0, p["d1.w"], p["d1.b"])
        x01 = ops.concat([x0, x1], axis=-1)
        x2 = conv_relu(ops, x01, p["d2.w"], p["d2.b"])
        x = ops.concat([x01, x2], axis=-1)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("densenet", ops, forward)


def build_inceptionv3(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("inceptionv3"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)
    p["inc.b1"] = pb.conv2d(16, 8, k=1)
    p["inc.b3"] = pb.conv2d(16, 8, k=3)
    p["inc.b5"] = pb.conv2d(16, 8, k=5)
    p["inc.bp"] = pb.conv2d(16, 8, k=1)
    p["proj.w"], p["proj.b"] = pb.conv2d(32, 32, k=1)
    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"])
        x = inception_block(ops, x, p["inc.b1"], p["inc.b3"], p["inc.b5"], p["inc.bp"])
        x = conv_relu(ops, x, p["proj.w"], p["proj.b"], padding=0)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("inceptionv3", ops, forward)


def build_inceptionv4(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("inceptionv4"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)
    # Two inception blocks chained
    for i in [1, 2]:
        p[f"inc{i}.b1"] = pb.conv2d(16 if i == 1 else 32, 8, k=1)
        p[f"inc{i}.b3"] = pb.conv2d(16 if i == 1 else 32, 8, k=3)
        p[f"inc{i}.b5"] = pb.conv2d(16 if i == 1 else 32, 8, k=5)
        p[f"inc{i}.bp"] = pb.conv2d(16 if i == 1 else 32, 8, k=1)
    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"])
        x = inception_block(ops, x, p["inc1.b1"], p["inc1.b3"], p["inc1.b5"], p["inc1.bp"])
        x = inception_block(ops, x, p["inc2.b1"], p["inc2.b3"], p["inc2.b5"], p["inc2.bp"])
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("inceptionv4", ops, forward)


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


def build_mnasnet(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("mnasnet"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)

    # One MBConv block (no SE)
    p["b.exp.w"], p["b.exp.b"] = pb.conv2d(16, 32, k=1)
    p["b.dw.w"], p["b.dw.b"] = pb.conv2d(32, 32, k=3, groups=32)
    p["b.proj.w"], p["b.proj.b"] = pb.conv2d(32, 16, k=1)
    p["head.w"], p["head.b"] = pb.linear(16, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"])
        x = _mbconv_block(ops, x, p, prefix="b", in_ch=16, out_ch=16, expansion=2, k=3, stride=1, se=False)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("mnasnet", ops, forward)


def build_efficientnet(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("efficientnet"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)

    # MBConv with SE
    p["b.exp.w"], p["b.exp.b"] = pb.conv2d(16, 64, k=1)
    p["b.dw.w"], p["b.dw.b"] = pb.conv2d(64, 64, k=3, groups=64)
    p["b.se.w1"], p["b.se.b1"] = pb.linear(64, 16)
    p["b.se.w2"], p["b.se.b2"] = pb.linear(16, 64)
    p["b.proj.w"], p["b.proj.b"] = pb.conv2d(64, 24, k=1)
    p["head.w"], p["head.b"] = pb.linear(24, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"])
        x = _mbconv_block(ops, x, p, prefix="b", in_ch=16, out_ch=24, expansion=4, k=3, stride=2, se=True)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("efficientnet", ops, forward)


def build_mobilenetv1(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("mobilenetv1"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)
    # depthwise + pointwise
    p["dw.w"], p["dw.b"] = pb.conv2d(16, 16, k=3, groups=16)
    p["pw.w"], p["pw.b"] = pb.conv2d(16, 32, k=1)
    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"], stride=2)
        x = depthwise_separable_conv(ops, x, p["dw.w"], p["dw.b"], p["pw.w"], p["pw.b"], stride=1)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("mobilenetv1", ops, forward)


def build_mobilenetv2(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("mobilenetv2"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)

    # Inverted residual: expand -> depthwise -> project
    p["b.exp.w"], p["b.exp.b"] = pb.conv2d(16, 48, k=1)
    p["b.dw.w"], p["b.dw.b"] = pb.conv2d(48, 48, k=3, groups=48)
    p["b.proj.w"], p["b.proj.b"] = pb.conv2d(48, 16, k=1)
    p["head.w"], p["head.b"] = pb.linear(16, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"], stride=2)
        y = conv_relu(ops, x, p["b.exp.w"], p["b.exp.b"], padding=0)
        y = ops.relu(ops.conv2d(y, p["b.dw.w"], p["b.dw.b"], padding=1, groups=48))
        y = ops.conv2d(y, p["b.proj.w"], p["b.proj.b"], padding=0)
        x = ops.relu(x + y)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("mobilenetv2", ops, forward)


def build_mobilenetv3(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("mobilenetv3"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)

    # Inverted residual with SE
    p["b.exp.w"], p["b.exp.b"] = pb.conv2d(16, 48, k=1)
    p["b.dw.w"], p["b.dw.b"] = pb.conv2d(48, 48, k=3, groups=48)
    p["b.se.w1"], p["b.se.b1"] = pb.linear(48, 12)
    p["b.se.w2"], p["b.se.b2"] = pb.linear(12, 48)
    p["b.proj.w"], p["b.proj.b"] = pb.conv2d(48, 16, k=1)
    p["head.w"], p["head.b"] = pb.linear(16, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"], stride=2)
        y = conv_relu(ops, x, p["b.exp.w"], p["b.exp.b"], padding=0)
        y = ops.relu(ops.conv2d(y, p["b.dw.w"], p["b.dw.b"], padding=1, groups=48))
        y = se_block(ops, y, p["b.se.w1"], p["b.se.b1"], p["b.se.w2"], p["b.se.b2"])
        y = ops.conv2d(y, p["b.proj.w"], p["b.proj.b"], padding=0)
        x = ops.relu(x + y)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("mobilenetv3", ops, forward)


def build_shufflenetv1(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("shufflenetv1"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)
    p["g1.w"], p["g1.b"] = pb.conv2d(16, 16, k=1, groups=4)
    p["dw.w"], p["dw.b"] = pb.conv2d(16, 16, k=3, groups=16)
    p["g2.w"], p["g2.b"] = pb.conv2d(16, 32, k=1, groups=4)
    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"], stride=2)
        y = conv_relu(ops, x, p["g1.w"], p["g1.b"], padding=0, groups=4)
        y = channel_shuffle_nhwc(ops, y, groups=4)
        y = ops.relu(ops.conv2d(y, p["dw.w"], p["dw.b"], padding=1, groups=16))
        y = ops.conv2d(y, p["g2.w"], p["g2.b"], padding=0, groups=4)
        x = ops.relu(y)
        logits = _classifier_head(ops, x, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("shufflenetv1", ops, forward)


def build_shufflenet_v2(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("shufflenet_v2"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)
    p["p1.w"], p["p1.b"] = pb.conv2d(16, 16, k=1)
    p["dw.w"], p["dw.b"] = pb.conv2d(16, 16, k=3, groups=16)
    p["p2.w"], p["p2.b"] = pb.conv2d(16, 32, k=1)
    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"], stride=2)
        y = conv_relu(ops, x, p["p1.w"], p["p1.b"], padding=0)
        y = ops.relu(ops.conv2d(y, p["dw.w"], p["dw.b"], padding=1, groups=16))
        y = conv_relu(ops, y, p["p2.w"], p["p2.b"], padding=0)
        logits = _classifier_head(ops, y, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("shufflenet_v2", ops, forward)


def build_xception(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("xception"))
    p: dict[str, Any] = {}
    p["stem.w"], p["stem.b"] = pb.conv2d(3, 16, k=3)
    p["dw1.w"], p["dw1.b"] = pb.conv2d(16, 16, k=3, groups=16)
    p["pw1.w"], p["pw1.b"] = pb.conv2d(16, 32, k=1)
    p["dw2.w"], p["dw2.b"] = pb.conv2d(32, 32, k=3, groups=32)
    p["pw2.w"], p["pw2.b"] = pb.conv2d(32, 32, k=1)
    p["head.w"], p["head.b"] = pb.linear(32, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = conv_relu(ops, inputs["image"], p["stem.w"], p["stem.b"], stride=2)
        y = depthwise_separable_conv(ops, x, p["dw1.w"], p["dw1.b"], p["pw1.w"], p["pw1.b"])
        y = depthwise_separable_conv(ops, y, p["dw2.w"], p["dw2.b"], p["pw2.w"], p["pw2.b"])
        logits = _classifier_head(ops, y, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("xception", ops, forward)


# -------------------------
# Detection models (raw heads only)
# -------------------------


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

    return ForwardModel(model_id, ops, forward)


def build_r_cnn(ops: Any) -> ForwardModel:
    return _build_rcnn_family(ops, "r_cnn", with_rpn=False, with_mask=False, cascade=1)


def build_fast_r_cnn(ops: Any) -> ForwardModel:
    return _build_rcnn_family(ops, "fast_r_cnn", with_rpn=False, with_mask=False, cascade=1)


def build_faster_r_cnn(ops: Any) -> ForwardModel:
    return _build_rcnn_family(ops, "faster_r_cnn", with_rpn=True, with_mask=False, cascade=1)


def build_cascade_r_cnn(ops: Any) -> ForwardModel:
    return _build_rcnn_family(ops, "cascade_r_cnn", with_rpn=True, with_mask=False, cascade=3)


def build_mask_rcnn(ops: Any) -> ForwardModel:
    return _build_rcnn_family(ops, "mask_rcnn", with_rpn=True, with_mask=True, cascade=1)


def _build_yolo_like(ops: Any, model_id: str, *, head_ch: int):
    pb = ParamBuilder(ops, seed=_seed(model_id))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["head.w"], p["head.b"] = pb.conv2d(64, head_ch, k=1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        _, _, feat = _simple_backbone_forward(ops, x, p)
        pred = ops.conv2d(feat, p["head.w"], p["head.b"], padding=0)
        return {"pred": pred}

    return ForwardModel(model_id, ops, forward)


def build_yolov1(ops: Any) -> ForwardModel:
    return _build_yolo_like(ops, "yolov1", head_ch=3 * (5 + NUM_CLASSES))


def build_yolov2(ops: Any) -> ForwardModel:
    return _build_yolo_like(ops, "yolov2", head_ch=5 * (5 + NUM_CLASSES))


def build_yolov3(ops: Any) -> ForwardModel:
    return _build_yolo_like(ops, "yolov3", head_ch=3 * (5 + NUM_CLASSES))


def build_yolov4(ops: Any) -> ForwardModel:
    return _build_yolo_like(ops, "yolov4", head_ch=3 * (5 + NUM_CLASSES))


def build_yolov5(ops: Any) -> ForwardModel:
    return _build_yolo_like(ops, "yolov5", head_ch=3 * (5 + NUM_CLASSES))


def build_ppyoloe(ops: Any) -> ForwardModel:
    return _build_yolo_like(ops, "ppyoloe", head_ch=3 * (5 + NUM_CLASSES))


def build_fpn(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("fpn"))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["l1.w"], p["l1.b"] = pb.conv2d(16, 32, k=1)
    p["l2.w"], p["l2.b"] = pb.conv2d(32, 32, k=1)
    p["l3.w"], p["l3.b"] = pb.conv2d(64, 32, k=1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x1, x2, x3 = _simple_backbone_forward(ops, inputs["image"], p)
        p1 = ops.conv2d(x1, p["l1.w"], p["l1.b"], padding=0)
        p2 = ops.conv2d(x2, p["l2.w"], p["l2.b"], padding=0)
        p3 = ops.conv2d(x3, p["l3.w"], p["l3.b"], padding=0)
        # simple top-down fusion
        p2 = p2 + ops.upsample2d_nearest(p3, scale=2)
        p1 = p1 + ops.upsample2d_nearest(p2, scale=2)
        return {"p1": p1, "p2": p2, "p3": p3}

    return ForwardModel("fpn", ops, forward)


def build_retinanet(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("retinanet"))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["neck.w"], p["neck.b"] = pb.conv2d(64, 32, k=1)
    p["cls.w"], p["cls.b"] = pb.conv2d(32, 3 * NUM_CLASSES, k=1)
    p["box.w"], p["box.b"] = pb.conv2d(32, 3 * 4, k=1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        _, _, feat = _simple_backbone_forward(ops, inputs["image"], p)
        feat = conv_relu(ops, feat, p["neck.w"], p["neck.b"], padding=0)
        cls = ops.conv2d(feat, p["cls.w"], p["cls.b"], padding=0)
        box = ops.conv2d(feat, p["box.w"], p["box.b"], padding=0)
        return {"cls": cls, "box": box}

    return ForwardModel("retinanet", ops, forward)


def build_fcos(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("fcos"))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["neck.w"], p["neck.b"] = pb.conv2d(64, 32, k=1)
    p["cls.w"], p["cls.b"] = pb.conv2d(32, NUM_CLASSES, k=1)
    p["reg.w"], p["reg.b"] = pb.conv2d(32, 4, k=1)
    p["ctr.w"], p["ctr.b"] = pb.conv2d(32, 1, k=1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        _, _, feat = _simple_backbone_forward(ops, inputs["image"], p)
        feat = conv_relu(ops, feat, p["neck.w"], p["neck.b"], padding=0)
        return {
            "cls": ops.conv2d(feat, p["cls.w"], p["cls.b"], padding=0),
            "reg": ops.conv2d(feat, p["reg.w"], p["reg.b"], padding=0),
            "centerness": ops.conv2d(feat, p["ctr.w"], p["ctr.b"], padding=0),
        }

    return ForwardModel("fcos", ops, forward)


def build_m2det(ops: Any) -> ForwardModel:
    # Minimal "multi-level" head over two scales.
    pb = ParamBuilder(ops, seed=_seed("m2det"))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["h2.w"], p["h2.b"] = pb.conv2d(32, 16, k=3)
    p["h3.w"], p["h3.b"] = pb.conv2d(64, 16, k=3)
    p["out.w"], p["out.b"] = pb.conv2d(32, 3 * (5 + NUM_CLASSES), k=1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        _, f2, f3 = _simple_backbone_forward(ops, inputs["image"], p)
        h2 = conv_relu(ops, f2, p["h2.w"], p["h2.b"])
        h3 = conv_relu(ops, f3, p["h3.w"], p["h3.b"])
        h3_up = ops.upsample2d_nearest(h3, scale=2)
        fused = ops.concat([h2, h3_up], axis=-1)
        pred = ops.conv2d(fused, p["out.w"], p["out.b"], padding=0)
        return {"pred": pred}

    return ForwardModel("m2det", ops, forward)


def build_efficientdet(ops: Any) -> ForwardModel:
    # Toy EfficientDet: backbone + BiFPN-like fusion + head
    pb = ParamBuilder(ops, seed=_seed("efficientdet"))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["p2.w"], p["p2.b"] = pb.conv2d(32, 32, k=1)
    p["p3.w"], p["p3.b"] = pb.conv2d(64, 32, k=1)
    p["head.w"], p["head.b"] = pb.conv2d(32, 3 * (5 + NUM_CLASSES), k=1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        _, f2, f3 = _simple_backbone_forward(ops, inputs["image"], p)
        p2 = ops.conv2d(f2, p["p2.w"], p["p2.b"], padding=0)
        p3 = ops.conv2d(f3, p["p3.w"], p["p3.b"], padding=0)
        fused = p2 + ops.upsample2d_nearest(p3, scale=2)
        pred = ops.conv2d(fused, p["head.w"], p["head.b"], padding=0)
        return {"pred": pred}

    return ForwardModel("efficientdet", ops, forward)


def build_cascade_rcnn_rs(ops: Any) -> ForwardModel:
    # Treat as a cascade variant.
    return _build_rcnn_family(ops, "cascade_rcnn_rs", with_rpn=True, with_mask=False, cascade=2)


def build_rt_detr(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("rt_detr"))
    p = _simple_backbone_params(pb, in_ch=3, base=16)

    # Project to token dim and run a single transformer block over flattened 4x4 = 16 tokens.
    p["proj.w"], p["proj.b"] = pb.conv2d(64, 64, k=1)
    # Transformer params (D=64)
    for k in ["wq", "wk", "wv", "wo"]:
        p[f"tr.{k}"], p[f"tr.b{k[-1]}"] = pb.linear(64, 64)
    p["tr.w1"], p["tr.b1"] = pb.linear(64, 128)
    p["tr.w2"], p["tr.b2"] = pb.linear(128, 64)
    # Output heads for fixed queries (use pooled token features)
    p["cls.w"], p["cls.b"] = pb.linear(64, NUM_CLASSES)
    p["box.w"], p["box.b"] = pb.linear(64, 4)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        _, _, feat = _simple_backbone_forward(ops, inputs["image"], p)
        feat = ops.conv2d(feat, p["proj.w"], p["proj.b"], padding=0)  # (B,4,4,64)
        b, h, w, c = (int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2]), int(feat.shape[3]))
        tokens = ops.reshape(feat, (b, h * w, c))

        # Map to expected transformer param keys
        tr_params = {
            "tr.wq": p["tr.wq"],
            "tr.bq": p["tr.bq"],
            "tr.wk": p["tr.wk"],
            "tr.bk": p["tr.bk"],
            "tr.wv": p["tr.wv"],
            "tr.bv": p["tr.bv"],
            "tr.wo": p["tr.wo"],
            "tr.bo": p["tr.bo"],
            "tr.w1": p["tr.w1"],
            "tr.b1": p["tr.b1"],
            "tr.w2": p["tr.w2"],
            "tr.b2": p["tr.b2"],
        }
        tokens = transformer_encoder_block(ops, tokens, tr_params, prefix="tr", num_heads=4)
        pooled = tokens[:, 0, :]  # use first token as "query"
        pooled = ops.reshape(pooled, (b, 64))
        cls = ops.linear(pooled, p["cls.w"], p["cls.b"])
        box = ops.linear(pooled, p["box.w"], p["box.b"])
        # Expand to (B, num_queries=10, ...)
        cls = ops.reshape(cls, (b, 1, NUM_CLASSES))
        box = ops.reshape(box, (b, 1, 4))
        return {"query_logits": cls, "query_boxes": box}

    return ForwardModel("rt_detr", ops, forward)


# -------------------------
# Segmentation models (logits only)
# -------------------------


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

    return ForwardModel(model_id, ops, forward)


def build_fcn(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "fcn", variant="fcn")


def build_deconvnet(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "deconvnet", variant="deconvnet")


def build_unet(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "unet", variant="unet")


def build_segnet(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "segnet", variant="segnet")


def build_enet(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "enet", variant="enet")


def build_fusionnet(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "fusionnet", variant="fusionnet")


def build_deeplabv1(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "deeplabv1", variant="deeplabv1")


def build_deeplabv2(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "deeplabv2", variant="deeplabv2")


def build_deeplabv3(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "deeplabv3", variant="deeplabv3")


def build_deeplabv3_plus(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "deeplabv3_plus", variant="deeplabv3_plus")


def build_gcn(ops: Any) -> ForwardModel:
    # Global convolution network: emulate large kernel conv.
    pb = ParamBuilder(ops, seed=_seed("gcn"))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["g.w"], p["g.b"] = pb.conv2d(64, 64, k=7)
    p["seg.w"], p["seg.b"] = pb.conv2d(64, NUM_SEG_CLASSES, k=1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        _, _, feat = _simple_backbone_forward(ops, inputs["image"], p)
        feat = conv_relu(ops, feat, p["g.w"], p["g.b"], padding=3)
        seg = ops.conv2d(feat, p["seg.w"], p["seg.b"], padding=0)
        seg = ops.upsample2d_nearest(seg, scale=8)
        return {"seg_logits": seg}

    return ForwardModel("gcn", ops, forward)


def build_exfuse(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "exfuse", variant="bisenet")


def build_dfn(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "dfn", variant="bisenet")


def build_bisenetv1(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "bisenetv1", variant="bisenet")


def build_bisenet_v2(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "bisenet_v2", variant="bisenet")


def build_rdfnet(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "rdfnet", variant="unet")


def build_rednet(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "rednet", variant="fusionnet")


def build_dfanet(ops: Any) -> ForwardModel:
    return _build_simple_segmentation(ops, "dfanet", variant="enet")


# -------------------------
# Vision Transformers
# -------------------------


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

    return ForwardModel(model_id, ops, forward)


def build_transformer(ops: Any) -> ForwardModel:
    return _vit_like(ops, "transformer", patch=4, embed_dim=48, depth=1, heads=4)


def build_vit(ops: Any) -> ForwardModel:
    return _vit_like(ops, "vit", patch=4, embed_dim=64, depth=2, heads=4)


def build_deit(ops: Any) -> ForwardModel:
    return _vit_like(ops, "deit", patch=4, embed_dim=64, depth=2, heads=4)


def build_t2t(ops: Any) -> ForwardModel:
    # tokens-to-token (toy): conv stem + smaller patch
    return _vit_like(ops, "t2t", patch=2, embed_dim=48, depth=2, heads=3)


def build_botnet(ops: Any) -> ForwardModel:
    # Toy BotNet: CNN backbone + one transformer block over spatial tokens.
    pb = ParamBuilder(ops, seed=_seed("botnet"))
    p = _simple_backbone_params(pb, in_ch=3, base=16)
    p["proj.w"], p["proj.b"] = pb.conv2d(64, 64, k=1)
    p["b0.wq"], p["b0.bq"] = pb.linear(64, 64)
    p["b0.wk"], p["b0.bk"] = pb.linear(64, 64)
    p["b0.wv"], p["b0.bv"] = pb.linear(64, 64)
    p["b0.wo"], p["b0.bo"] = pb.linear(64, 64)
    p["b0.w1"], p["b0.b1"] = pb.linear(64, 128)
    p["b0.w2"], p["b0.b2"] = pb.linear(128, 64)
    p["head.w"], p["head.b"] = pb.linear(64, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        _, _, feat = _simple_backbone_forward(ops, inputs["image"], p)
        feat = ops.conv2d(feat, p["proj.w"], p["proj.b"], padding=0)
        b, h, w, c = (int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2]), int(feat.shape[3]))
        tokens = ops.reshape(feat, (b, h * w, c))
        tokens = transformer_encoder_block(ops, tokens, p, prefix="b0", num_heads=4)
        pooled = tokens[:, 0, :]
        pooled = ops.reshape(pooled, (b, c))
        logits = ops.linear(pooled, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("botnet", ops, forward)


def build_tnt(ops: Any) -> ForwardModel:
    # Toy TNT: two transformer blocks in series with different heads.
    return _vit_like(ops, "tnt", patch=4, embed_dim=64, depth=2, heads=2)


def build_mae(ops: Any) -> ForwardModel:
    # Toy MAE: return latent tokens and a linear reconstruction of patches.
    pb = ParamBuilder(ops, seed=_seed("mae"))
    p: dict[str, Any] = {}
    patch = 4
    embed_dim = 48
    p["pe.w"], p["pe.b"] = pb.conv2d(3, embed_dim, k=patch)
    p["b0.wq"], p["b0.bq"] = pb.linear(embed_dim, embed_dim)
    p["b0.wk"], p["b0.bk"] = pb.linear(embed_dim, embed_dim)
    p["b0.wv"], p["b0.bv"] = pb.linear(embed_dim, embed_dim)
    p["b0.wo"], p["b0.bo"] = pb.linear(embed_dim, embed_dim)
    p["b0.w1"], p["b0.b1"] = pb.linear(embed_dim, embed_dim * 2)
    p["b0.w2"], p["b0.b2"] = pb.linear(embed_dim * 2, embed_dim)
    p["recon.w"], p["recon.b"] = pb.linear(embed_dim, patch * patch * 3)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        tokens, _ = patch_embed(ops, inputs["image"], p["pe.w"], p["pe.b"], patch=patch, embed_dim=embed_dim)
        tokens = transformer_encoder_block(ops, tokens, p, prefix="b0", num_heads=4)
        recon = ops.linear(tokens, p["recon.w"], p["recon.b"])
        return {"latent": tokens, "recon_patches": recon}

    return ForwardModel("mae", ops, forward)


def build_pvt(ops: Any) -> ForwardModel:
    # Toy PVT: 2-stage pyramid attention by downsampling tokens.
    pb = ParamBuilder(ops, seed=_seed("pvt"))
    p: dict[str, Any] = {}
    p["pe.w"], p["pe.b"] = pb.conv2d(3, 48, k=4)
    # stage1
    p["s1.wq"], p["s1.bq"] = pb.linear(48, 48)
    p["s1.wk"], p["s1.bk"] = pb.linear(48, 48)
    p["s1.wv"], p["s1.bv"] = pb.linear(48, 48)
    p["s1.wo"], p["s1.bo"] = pb.linear(48, 48)
    p["s1.w1"], p["s1.b1"] = pb.linear(48, 96)
    p["s1.w2"], p["s1.b2"] = pb.linear(96, 48)
    # stage2
    p["s2.wq"], p["s2.bq"] = pb.linear(48, 48)
    p["s2.wk"], p["s2.bk"] = pb.linear(48, 48)
    p["s2.wv"], p["s2.bv"] = pb.linear(48, 48)
    p["s2.wo"], p["s2.bo"] = pb.linear(48, 48)
    p["s2.w1"], p["s2.b1"] = pb.linear(48, 96)
    p["s2.w2"], p["s2.b2"] = pb.linear(96, 48)
    p["head.w"], p["head.b"] = pb.linear(48, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        tokens, (gh, gw) = patch_embed(ops, inputs["image"], p["pe.w"], p["pe.b"], patch=4, embed_dim=48)
        tokens = transformer_encoder_block(ops, tokens, p, prefix="s1", num_heads=3)
        # Downsample token grid by 2 (avg pool on grid then re-flatten)
        b = int(tokens.shape[0])
        tokens2d = ops.reshape(tokens, (b, gh, gw, 48))
        tokens2d = ops.avg_pool2d(tokens2d, kernel=2, stride=2)
        b, gh2, gw2, _ = (int(tokens2d.shape[0]), int(tokens2d.shape[1]), int(tokens2d.shape[2]), int(tokens2d.shape[3]))
        tokens = ops.reshape(tokens2d, (b, gh2 * gw2, 48))
        tokens = transformer_encoder_block(ops, tokens, p, prefix="s2", num_heads=3)
        pooled = ops.reduce_mean(tokens, axis=1)
        logits = ops.linear(pooled, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("pvt", ops, forward)


def build_swin_transformer(ops: Any) -> ForwardModel:
    # Toy Swin: window attention over 8x8 patch grid with window=4.
    pb = ParamBuilder(ops, seed=_seed("swin_transformer"))
    p: dict[str, Any] = {}
    embed_dim = 48
    p["pe.w"], p["pe.b"] = pb.conv2d(3, embed_dim, k=4)
    p["b0.wq"], p["b0.bq"] = pb.linear(embed_dim, embed_dim)
    p["b0.wk"], p["b0.bk"] = pb.linear(embed_dim, embed_dim)
    p["b0.wv"], p["b0.bv"] = pb.linear(embed_dim, embed_dim)
    p["b0.wo"], p["b0.bo"] = pb.linear(embed_dim, embed_dim)
    p["b0.w1"], p["b0.b1"] = pb.linear(embed_dim, embed_dim * 2)
    p["b0.w2"], p["b0.b2"] = pb.linear(embed_dim * 2, embed_dim)
    p["head.w"], p["head.b"] = pb.linear(embed_dim, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        tokens, (gh, gw) = patch_embed(ops, inputs["image"], p["pe.w"], p["pe.b"], patch=4, embed_dim=embed_dim)
        b = int(tokens.shape[0])
        grid = ops.reshape(tokens, (b, gh, gw, embed_dim))  # (B,8,8,C)

        # Partition into windows of 4x4 => (B, 2, 4, 2, 4, C) -> (B*4, 16, C)
        grid = ops.reshape(grid, (b, 2, 4, 2, 4, embed_dim))
        grid = ops.transpose(grid, (0, 1, 3, 2, 4, 5))
        win = ops.reshape(grid, (b * 4, 16, embed_dim))

        win = transformer_encoder_block(ops, win, p, prefix="b0", num_heads=3)

        # Merge windows back
        grid = ops.reshape(win, (b, 2, 2, 4, 4, embed_dim))
        grid = ops.transpose(grid, (0, 1, 3, 2, 4, 5))
        grid = ops.reshape(grid, (b, gh, gw, embed_dim))

        tokens = ops.reshape(grid, (b, gh * gw, embed_dim))
        pooled = ops.reduce_mean(tokens, axis=1)
        logits = ops.linear(pooled, p["head.w"], p["head.b"])
        return {"logits": logits}

    return ForwardModel("swin_transformer", ops, forward)


# -------------------------
# GANs
# -------------------------


def build_gan(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("gan"))
    p: dict[str, Any] = {}
    p["g1.w"], p["g1.b"] = pb.linear(64, 128)
    p["g2.w"], p["g2.b"] = pb.linear(128, 32 * 32 * 3)
    p["d1.w"], p["d1.b"] = pb.linear(32 * 32 * 3, 128)
    p["d2.w"], p["d2.b"] = pb.linear(128, 1)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        z = inputs["z"]
        h = ops.relu(ops.linear(z, p["g1.w"], p["g1.b"]))
        img = ops.linear(h, p["g2.w"], p["g2.b"])
        # reshape to NHWC
        b = int(img.shape[0])
        img = ops.reshape(img, (b, 32, 32, 3))

        flat = ops.reshape(img, (b, 32 * 32 * 3))
        dh = ops.relu(ops.linear(flat, p["d1.w"], p["d1.b"]))
        score = ops.linear(dh, p["d2.w"], p["d2.b"])
        return {"gen_image": img, "disc_score": score}

    return ForwardModel("gan", ops, forward)


def build_pix2pix(ops: Any) -> ForwardModel:
    # Toy pix2pix: small U-Net-ish generator + patch discriminator.
    pb = ParamBuilder(ops, seed=_seed("pix2pix"))
    p: dict[str, Any] = {}
    p["e1.w"], p["e1.b"] = pb.conv2d(3, 16, k=3)
    p["e2.w"], p["e2.b"] = pb.conv2d(16, 32, k=3)
    p["d1.w"], p["d1.b"] = pb.conv2d(32, 16, k=3)
    p["out.w"], p["out.b"] = pb.conv2d(16, 3, k=1)
    p["disc.w"], p["disc.b"] = pb.conv2d(3, 1, k=3)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        e1 = conv_relu(ops, x, p["e1.w"], p["e1.b"])
        e2 = conv_relu(ops, ops.max_pool2d(e1, kernel=2, stride=2), p["e2.w"], p["e2.b"])
        d = ops.upsample2d_nearest(e2, scale=2)
        d = conv_relu(ops, d, p["d1.w"], p["d1.b"])
        out = ops.conv2d(d, p["out.w"], p["out.b"], padding=0)
        disc = ops.conv2d(out, p["disc.w"], p["disc.b"], padding=1)
        return {"gen_image": out, "patch_score": disc}

    return ForwardModel("pix2pix", ops, forward)


def build_cyclegan(ops: Any) -> ForwardModel:
    # Toy CycleGAN: ResNet-ish generator and discriminator.
    pb = ParamBuilder(ops, seed=_seed("cyclegan"))
    p: dict[str, Any] = {}
    p["g.stem.w"], p["g.stem.b"] = pb.conv2d(3, 16, k=3)
    p["g.b1.w1"], p["g.b1.b1"] = pb.conv2d(16, 16, k=3)
    p["g.b1.w2"], p["g.b1.b2"] = pb.conv2d(16, 16, k=3)
    p["g.out.w"], p["g.out.b"] = pb.conv2d(16, 3, k=1)

    p["d.stem.w"], p["d.stem.b"] = pb.conv2d(3, 16, k=3)
    p["d.out.w"], p["d.out.b"] = pb.conv2d(16, 1, k=3)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["image"]
        g = conv_relu(ops, x, p["g.stem.w"], p["g.stem.b"])
        g = residual_block(ops, g, p["g.b1.w1"], p["g.b1.b1"], p["g.b1.w2"], p["g.b1.b2"])
        g = ops.conv2d(g, p["g.out.w"], p["g.out.b"], padding=0)

        d = conv_relu(ops, g, p["d.stem.w"], p["d.stem.b"])
        score = ops.conv2d(d, p["d.out.w"], p["d.out.b"], padding=1)
        return {"gen_image": g, "patch_score": score}

    return ForwardModel("cyclegan", ops, forward)


# -------------------------
# Graph models
# -------------------------


def build_node2vec(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("node2vec"))
    p: dict[str, Any] = {}
    p["emb"] = pb.embedding(16, 16)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        src = inputs["src"]
        dst = inputs["dst"]
        e_src = ops.gather(p["emb"], src, axis=0)
        e_dst = ops.gather(p["emb"], dst, axis=0)
        score = ops.reduce_sum(e_src * e_dst, axis=-1, keepdims=True)
        return {"score": score}

    return ForwardModel("node2vec", ops, forward)


def build_line(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("line"))
    p: dict[str, Any] = {}
    p["emb1"] = pb.embedding(16, 16)
    p["emb2"] = pb.embedding(16, 16)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        src = inputs["src"]
        dst = inputs["dst"]
        e1 = ops.gather(p["emb1"], src, axis=0)
        e2 = ops.gather(p["emb2"], dst, axis=0)
        score = ops.reduce_sum(e1 * e2, axis=-1, keepdims=True)
        return {"score": score}

    return ForwardModel("line", ops, forward)


def build_metapath2vec(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("metapath2vec"))
    p: dict[str, Any] = {}
    p["emb"] = pb.embedding(16, 16)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        src = inputs["src"]
        dst = inputs["dst"]
        e_src = ops.gather(p["emb"], src, axis=0)
        e_dst = ops.gather(p["emb"], dst, axis=0)
        score = ops.reduce_sum(e_src * e_dst, axis=-1, keepdims=True)
        return {"score": score}

    return ForwardModel("metapath2vec", ops, forward)


def build_sdne(ops: Any) -> ForwardModel:
    pb = ParamBuilder(ops, seed=_seed("sdne"))
    p: dict[str, Any] = {}
    p["w1"], p["b1"] = pb.linear(16, 16)
    p["w2"], p["b2"] = pb.linear(16, 16)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["x"]  # (N, F=16)
        h = ops.relu(ops.linear(x, p["w1"], p["b1"]))
        recon = ops.sigmoid(ops.linear(h, p["w2"], p["b2"]))
        return {"recon": recon}

    return ForwardModel("sdne", ops, forward)


def build_graph_neural_networks(ops: Any) -> ForwardModel:
    # Tiny GCN
    pb = ParamBuilder(ops, seed=_seed("graph_neural_networks"))
    p: dict[str, Any] = {}
    p["w1"], p["b1"] = pb.linear(16, 16)
    p["w2"], p["b2"] = pb.linear(16, NUM_CLASSES)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["x"]  # (N, F)
        adj = inputs["adj"]  # (N, N)
        # message passing: A @ X
        h = ops.matmul(adj, x)
        h = ops.relu(ops.linear(h, p["w1"], p["b1"]))
        logits = ops.linear(h, p["w2"], p["b2"])
        return {"node_logits": logits}

    return ForwardModel("graph_neural_networks", ops, forward)


def build_a_survey_on_graph_diffusion_models(ops: Any) -> ForwardModel:
    # Toy "graph diffusion": one denoising step on node features.
    pb = ParamBuilder(ops, seed=_seed("a_survey_on_graph_diffusion_models"))
    p: dict[str, Any] = {}
    p["w1"], p["b1"] = pb.linear(16, 16)
    p["w2"], p["b2"] = pb.linear(16, 16)

    def forward(inputs: dict[str, Any]) -> dict[str, Any]:
        x = inputs["x"]
        adj = inputs["adj"]
        h = ops.matmul(adj, x)
        h = ops.gelu(ops.linear(h, p["w1"], p["b1"]))
        noise_pred = ops.linear(h, p["w2"], p["b2"])
        return {"noise_pred": noise_pred}

    return ForwardModel("a_survey_on_graph_diffusion_models", ops, forward)


# -------------------------
# Registry mapping
# -------------------------


BUILDERS: dict[str, Callable[[Any], ForwardModel]] = {
    # classification
    "bp": build_bp,
    "zfnet": build_zfnet,
    "vgg": build_vgg,
    "googlenet": build_googlenet,
    "nin": build_nin,
    "resnet": build_resnet,
    "resnext": build_resnext,
    "inceptionv3": build_inceptionv3,
    "inceptionv4": build_inceptionv4,
    "mnasnet": build_mnasnet,
    "senet": build_senet,
    "densenet": build_densenet,
    "efficientnet": build_efficientnet,
    "mobilenetv1": build_mobilenetv1,
    "mobilenetv2": build_mobilenetv2,
    "mobilenetv3": build_mobilenetv3,
    "shufflenetv1": build_shufflenetv1,
    "shufflenet_v2": build_shufflenet_v2,
    "xception": build_xception,
    # detection
    "r_cnn": build_r_cnn,
    "fast_r_cnn": build_fast_r_cnn,
    "faster_r_cnn": build_faster_r_cnn,
    "cascade_r_cnn": build_cascade_r_cnn,
    "yolov1": build_yolov1,
    "yolov2": build_yolov2,
    "yolov3": build_yolov3,
    "yolov4": build_yolov4,
    "yolov5": build_yolov5,
    "ppyoloe": build_ppyoloe,
    "rt_detr": build_rt_detr,
    "fpn": build_fpn,
    "retinanet": build_retinanet,
    "fcos": build_fcos,
    "mask_rcnn": build_mask_rcnn,
    "m2det": build_m2det,
    "efficientdet": build_efficientdet,
    "cascade_rcnn_rs": build_cascade_rcnn_rs,
    # segmentation
    "fcn": build_fcn,
    "deconvnet": build_deconvnet,
    "unet": build_unet,
    "segnet": build_segnet,
    "enet": build_enet,
    "fusionnet": build_fusionnet,
    "deeplabv1": build_deeplabv1,
    "deeplabv2": build_deeplabv2,
    "deeplabv3": build_deeplabv3,
    "deeplabv3_plus": build_deeplabv3_plus,
    "gcn": build_gcn,
    "exfuse": build_exfuse,
    "dfn": build_dfn,
    "bisenetv1": build_bisenetv1,
    "bisenet_v2": build_bisenet_v2,
    "rdfnet": build_rdfnet,
    "rednet": build_rednet,
    "dfanet": build_dfanet,
    # transformers
    "transformer": build_transformer,
    "vit": build_vit,
    "t2t": build_t2t,
    "botnet": build_botnet,
    "tnt": build_tnt,
    "mae": build_mae,
    "pvt": build_pvt,
    "swin_transformer": build_swin_transformer,
    "deit": build_deit,
    # GAN
    "gan": build_gan,
    "pix2pix": build_pix2pix,
    "cyclegan": build_cyclegan,
    # graph
    "node2vec": build_node2vec,
    "line": build_line,
    "sdne": build_sdne,
    "metapath2vec": build_metapath2vec,
    "graph_neural_networks": build_graph_neural_networks,
    "a_survey_on_graph_diffusion_models": build_a_survey_on_graph_diffusion_models,
}
