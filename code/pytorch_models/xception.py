from __future__ import annotations

from .base import PyTorchModelBase


class Xception(PyTorchModelBase):
    model_id = "xception"


MODEL_ID = "xception"
MODEL_CLASS = Xception
