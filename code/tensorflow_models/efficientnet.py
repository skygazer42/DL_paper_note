from __future__ import annotations

from .base import TensorFlowModelBase


class EfficientNet(TensorFlowModelBase):
    model_id = "efficientnet"


MODEL_ID = "efficientnet"
MODEL_CLASS = EfficientNet
