from __future__ import annotations

from .base import PyTorchModelBase


class SDNE(PyTorchModelBase):
    model_id = "sdne"


MODEL_ID = "sdne"
MODEL_CLASS = SDNE
