from __future__ import annotations

from .base import PyTorchModelBase


class FPN(PyTorchModelBase):
    model_id = "fpn"


MODEL_ID = "fpn"
MODEL_CLASS = FPN
