from __future__ import annotations

from .base import PyTorchModelBase


class BiSeNetV2(PyTorchModelBase):
    model_id = "bisenet_v2"


MODEL_ID = "bisenet_v2"
MODEL_CLASS = BiSeNetV2
