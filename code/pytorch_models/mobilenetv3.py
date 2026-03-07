from __future__ import annotations

from .base import PyTorchModelBase


class MobileNetV3(PyTorchModelBase):
    model_id = "mobilenetv3"


MODEL_ID = "mobilenetv3"
MODEL_CLASS = MobileNetV3
