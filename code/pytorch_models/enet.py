from __future__ import annotations

from .base import PyTorchModelBase


class ENet(PyTorchModelBase):
    model_id = "enet"


MODEL_ID = "enet"
MODEL_CLASS = ENet
