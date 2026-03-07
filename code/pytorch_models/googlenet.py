from __future__ import annotations

from .base import PyTorchModelBase


class GoogleNet(PyTorchModelBase):
    model_id = "googlenet"


MODEL_ID = "googlenet"
MODEL_CLASS = GoogleNet
