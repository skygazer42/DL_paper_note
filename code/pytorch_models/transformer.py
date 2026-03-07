from __future__ import annotations

from .base import PyTorchModelBase


class Transformer(PyTorchModelBase):
    model_id = "transformer"


MODEL_ID = "transformer"
MODEL_CLASS = Transformer
