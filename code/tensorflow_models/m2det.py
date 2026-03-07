from __future__ import annotations

from .base import TensorFlowModelBase


class M2Det(TensorFlowModelBase):
    model_id = "m2det"


MODEL_ID = "m2det"
MODEL_CLASS = M2Det
