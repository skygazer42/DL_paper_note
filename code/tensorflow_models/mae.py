from __future__ import annotations

from .base import TensorFlowModelBase


class MAE(TensorFlowModelBase):
    model_id = "mae"


MODEL_ID = "mae"
MODEL_CLASS = MAE
