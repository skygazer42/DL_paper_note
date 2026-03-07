from __future__ import annotations

from .base import TensorFlowModelBase


class MnasNet(TensorFlowModelBase):
    model_id = "mnasnet"


MODEL_ID = "mnasnet"
MODEL_CLASS = MnasNet
