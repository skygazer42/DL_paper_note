from __future__ import annotations

from .base import PyTorchModelBase


class SeNet(PyTorchModelBase):
    model_id = "senet"


MODEL_ID = "senet"
MODEL_CLASS = SeNet
