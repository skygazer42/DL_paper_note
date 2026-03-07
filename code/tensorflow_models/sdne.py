from __future__ import annotations

from .base import TensorFlowModelBase


class SDNE(TensorFlowModelBase):
    model_id = "sdne"


MODEL_ID = "sdne"
MODEL_CLASS = SDNE
