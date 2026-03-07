from __future__ import annotations

from .base import PyTorchModelBase


class ResNet(PyTorchModelBase):
    model_id = "resnet"


MODEL_ID = "resnet"
MODEL_CLASS = ResNet
