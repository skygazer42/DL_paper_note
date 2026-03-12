
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
        # 参数初始化固定随机种子，这样不同机器/重复运行得到同样的 toy 权重。
        self.rng = np.random.default_rng(self.seed)

    def _randn(self, shape: tuple[int, ...], *, scale: float) -> Any:
        arr = (self.rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32, copy=False)
        return self.backend.asarray(arr)

    def zeros(self, shape: tuple[int, ...]) -> Any:
        arr = np.zeros(shape, dtype=np.float32)
        return self.backend.asarray(arr)

    def conv2d(self, in_ch: int, out_ch: int, *, k: int = 3, groups: int = 1) -> tuple[Any, Any]:
        # 用 fan-in 控制随机权重的尺度，避免 toy forward 一开始就数值爆炸。
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
