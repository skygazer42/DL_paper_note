from __future__ import annotations

from .base import TensorFlowModelBase


class LINE(TensorFlowModelBase):
    model_id = "line"


MODEL_ID = "line"
MODEL_CLASS = LINE
