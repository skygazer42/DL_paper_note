from __future__ import annotations

from .base import PyTorchModelBase


class DeepLabv3(PyTorchModelBase):
    model_id = "deeplabv3"


MODEL_ID = "deeplabv3"
MODEL_CLASS = DeepLabv3
