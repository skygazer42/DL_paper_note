from __future__ import annotations

from .base import TensorFlowModelBase


class FCOS(TensorFlowModelBase):
    model_id = "fcos"


MODEL_ID = "fcos"
MODEL_CLASS = FCOS
