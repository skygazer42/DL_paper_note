from __future__ import annotations

from .base import TensorFlowModelBase


class ShuffleNetV2(TensorFlowModelBase):
    model_id = "shufflenet_v2"


MODEL_ID = "shufflenet_v2"
MODEL_CLASS = ShuffleNetV2
