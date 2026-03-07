from __future__ import annotations

from .base import PyTorchModelBase


class TnT(PyTorchModelBase):
    model_id = "tnt"


MODEL_ID = "tnt"
MODEL_CLASS = TnT
