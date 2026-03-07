from __future__ import annotations

from .base import TensorFlowModelBase


class PVT(TensorFlowModelBase):
    model_id = "pvt"


MODEL_ID = "pvt"
MODEL_CLASS = PVT
