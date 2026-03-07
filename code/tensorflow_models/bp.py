from __future__ import annotations

from .base import TensorFlowModelBase


class BP(TensorFlowModelBase):
    model_id = "bp"


MODEL_ID = "bp"
MODEL_CLASS = BP
