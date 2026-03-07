from __future__ import annotations

from .base import PyTorchModelBase


class MobileNetV2(PyTorchModelBase):
    model_id = "mobilenetv2"


MODEL_ID = "mobilenetv2"
MODEL_CLASS = MobileNetV2
