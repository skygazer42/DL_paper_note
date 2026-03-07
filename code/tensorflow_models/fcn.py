from __future__ import annotations

from .base import TensorFlowModelBase


class FCN(TensorFlowModelBase):
    model_id = "fcn"


MODEL_ID = "fcn"
MODEL_CLASS = FCN
