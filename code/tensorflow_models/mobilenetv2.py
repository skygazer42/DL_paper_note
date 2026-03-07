from __future__ import annotations

from .base import TensorFlowModelBase


class MobileNetV2(TensorFlowModelBase):
    model_id = "mobilenetv2"


MODEL_ID = "mobilenetv2"
MODEL_CLASS = MobileNetV2
