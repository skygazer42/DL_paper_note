from __future__ import annotations

from .base import TensorFlowModelBase


class MobileNetv1(TensorFlowModelBase):
    model_id = "mobilenetv1"


MODEL_ID = "mobilenetv1"
MODEL_CLASS = MobileNetv1
