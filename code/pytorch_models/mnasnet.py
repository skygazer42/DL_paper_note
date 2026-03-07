from __future__ import annotations

from .base import PyTorchModelBase


class MnasNet(PyTorchModelBase):
    model_id = "mnasnet"


MODEL_ID = "mnasnet"
MODEL_CLASS = MnasNet
