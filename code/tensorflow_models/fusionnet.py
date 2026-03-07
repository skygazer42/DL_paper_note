from __future__ import annotations

from .base import TensorFlowModelBase


class FusionNet(TensorFlowModelBase):
    model_id = "fusionnet"


MODEL_ID = "fusionnet"
MODEL_CLASS = FusionNet
