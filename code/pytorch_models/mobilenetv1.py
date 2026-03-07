from __future__ import annotations

from .base import PyTorchModelBase


class MobileNetv1(PyTorchModelBase):
    model_id = "mobilenetv1"


MODEL_ID = "mobilenetv1"
MODEL_CLASS = MobileNetv1
