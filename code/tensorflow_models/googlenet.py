from __future__ import annotations

from .base import TensorFlowModelBase


class GoogleNet(TensorFlowModelBase):
    model_id = "googlenet"


MODEL_ID = "googlenet"
MODEL_CLASS = GoogleNet
