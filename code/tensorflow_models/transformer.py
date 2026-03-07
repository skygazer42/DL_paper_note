from __future__ import annotations

from .base import TensorFlowModelBase


class Transformer(TensorFlowModelBase):
    model_id = "transformer"


MODEL_ID = "transformer"
MODEL_CLASS = Transformer
