from __future__ import annotations

from .base import PyTorchModelBase


class BiSeNetv1(PyTorchModelBase):
    model_id = "bisenetv1"


MODEL_ID = "bisenetv1"
MODEL_CLASS = BiSeNetv1
