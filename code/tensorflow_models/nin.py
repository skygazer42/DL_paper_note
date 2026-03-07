from __future__ import annotations

from .base import TensorFlowModelBase


class NiN(TensorFlowModelBase):
    model_id = "nin"


MODEL_ID = "nin"
MODEL_CLASS = NiN
