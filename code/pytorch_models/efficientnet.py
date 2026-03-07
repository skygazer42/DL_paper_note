from __future__ import annotations

from .base import PyTorchModelBase


class EfficientNet(PyTorchModelBase):
    model_id = "efficientnet"


MODEL_ID = "efficientnet"
MODEL_CLASS = EfficientNet
