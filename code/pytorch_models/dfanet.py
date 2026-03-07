from __future__ import annotations

from .base import PyTorchModelBase


class DFANet(PyTorchModelBase):
    model_id = "dfanet"


MODEL_ID = "dfanet"
MODEL_CLASS = DFANet
