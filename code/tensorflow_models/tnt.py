from __future__ import annotations

from .base import TensorFlowModelBase


class TnT(TensorFlowModelBase):
    model_id = "tnt"


MODEL_ID = "tnt"
MODEL_CLASS = TnT
