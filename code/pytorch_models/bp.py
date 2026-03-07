from __future__ import annotations

from .base import PyTorchModelBase


class BP(PyTorchModelBase):
    model_id = "bp"


MODEL_ID = "bp"
MODEL_CLASS = BP
