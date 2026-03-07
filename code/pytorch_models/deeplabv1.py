from __future__ import annotations

from .base import PyTorchModelBase


class DeepLabv1(PyTorchModelBase):
    model_id = "deeplabv1"


MODEL_ID = "deeplabv1"
MODEL_CLASS = DeepLabv1
