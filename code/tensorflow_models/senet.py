from __future__ import annotations

from .base import TensorFlowModelBase


class SeNet(TensorFlowModelBase):
    model_id = "senet"


MODEL_ID = "senet"
MODEL_CLASS = SeNet
