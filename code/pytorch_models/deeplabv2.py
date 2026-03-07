from __future__ import annotations

from .base import PyTorchModelBase


class DeepLabv2(PyTorchModelBase):
    model_id = "deeplabv2"


MODEL_ID = "deeplabv2"
MODEL_CLASS = DeepLabv2
