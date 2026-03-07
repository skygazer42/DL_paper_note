from __future__ import annotations

from .base import PyTorchModelBase


class RedNet(PyTorchModelBase):
    model_id = "rednet"


MODEL_ID = "rednet"
MODEL_CLASS = RedNet
