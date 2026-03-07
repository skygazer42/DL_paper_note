from __future__ import annotations

from .base import PyTorchModelBase


class LINE(PyTorchModelBase):
    model_id = "line"


MODEL_ID = "line"
MODEL_CLASS = LINE
