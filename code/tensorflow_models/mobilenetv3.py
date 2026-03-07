from __future__ import annotations

from .base import TensorFlowModelBase


class MobileNetV3(TensorFlowModelBase):
    model_id = "mobilenetv3"


MODEL_ID = "mobilenetv3"
MODEL_CLASS = MobileNetV3
