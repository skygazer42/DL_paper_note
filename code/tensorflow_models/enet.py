from __future__ import annotations

from .base import TensorFlowModelBase


class ENet(TensorFlowModelBase):
    model_id = "enet"


MODEL_ID = "enet"
MODEL_CLASS = ENet
