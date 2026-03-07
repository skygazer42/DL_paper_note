from __future__ import annotations

from .base import PyTorchModelBase


class EfficientDet(PyTorchModelBase):
    model_id = "efficientdet"


MODEL_ID = "efficientdet"
MODEL_CLASS = EfficientDet
