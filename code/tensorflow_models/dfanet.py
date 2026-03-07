from __future__ import annotations

from .base import TensorFlowModelBase


class DFANet(TensorFlowModelBase):
    model_id = "dfanet"


MODEL_ID = "dfanet"
MODEL_CLASS = DFANet
