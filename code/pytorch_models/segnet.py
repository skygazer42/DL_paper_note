from __future__ import annotations

from .base import PyTorchModelBase


class SegNet(PyTorchModelBase):
    model_id = "segnet"


MODEL_ID = "segnet"
MODEL_CLASS = SegNet
