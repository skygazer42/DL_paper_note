from __future__ import annotations

from .base import TensorFlowModelBase


class Xception(TensorFlowModelBase):
    model_id = "xception"


MODEL_ID = "xception"
MODEL_CLASS = Xception
