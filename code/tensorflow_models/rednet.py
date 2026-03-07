from __future__ import annotations

from .base import TensorFlowModelBase


class RedNet(TensorFlowModelBase):
    model_id = "rednet"


MODEL_ID = "rednet"
MODEL_CLASS = RedNet
