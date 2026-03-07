from __future__ import annotations

from .base import TensorFlowModelBase


class Deit(TensorFlowModelBase):
    model_id = "deit"


MODEL_ID = "deit"
MODEL_CLASS = Deit
