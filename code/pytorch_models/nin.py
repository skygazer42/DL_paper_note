from __future__ import annotations

from .base import PyTorchModelBase


class NiN(PyTorchModelBase):
    model_id = "nin"


MODEL_ID = "nin"
MODEL_CLASS = NiN
