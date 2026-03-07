from __future__ import annotations

from .base import TensorFlowModelBase


class SegNet(TensorFlowModelBase):
    model_id = "segnet"


MODEL_ID = "segnet"
MODEL_CLASS = SegNet
