from __future__ import annotations

from .base import PyTorchModelBase


class RetinaNet(PyTorchModelBase):
    model_id = "retinanet"


MODEL_ID = "retinanet"
MODEL_CLASS = RetinaNet
