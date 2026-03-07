from __future__ import annotations

from .base import PyTorchModelBase


class VGG(PyTorchModelBase):
    model_id = "vgg"


MODEL_ID = "vgg"
MODEL_CLASS = VGG
