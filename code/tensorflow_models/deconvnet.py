from __future__ import annotations

from .base import TensorFlowModelBase


class DeconvNet(TensorFlowModelBase):
    model_id = "deconvnet"


MODEL_ID = "deconvnet"
MODEL_CLASS = DeconvNet
