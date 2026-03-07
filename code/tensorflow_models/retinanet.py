from __future__ import annotations

from .base import TensorFlowModelBase


class RetinaNet(TensorFlowModelBase):
    model_id = "retinanet"


MODEL_ID = "retinanet"
MODEL_CLASS = RetinaNet
