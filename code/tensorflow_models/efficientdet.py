from __future__ import annotations

from .base import TensorFlowModelBase


class EfficientDet(TensorFlowModelBase):
    model_id = "efficientdet"


MODEL_ID = "efficientdet"
MODEL_CLASS = EfficientDet
