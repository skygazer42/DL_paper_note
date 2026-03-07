from __future__ import annotations

from .base import PyTorchModelBase


class DeepLabv3Plus(PyTorchModelBase):
    model_id = "deeplabv3_plus"


MODEL_ID = "deeplabv3_plus"
MODEL_CLASS = DeepLabv3Plus
