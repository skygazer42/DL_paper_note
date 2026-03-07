from __future__ import annotations

from .base import TensorFlowModelBase


class FPN(TensorFlowModelBase):
    model_id = "fpn"


MODEL_ID = "fpn"
MODEL_CLASS = FPN
