from __future__ import annotations

from .base import PyTorchModelBase


class T2T(PyTorchModelBase):
    model_id = "t2t"


MODEL_ID = "t2t"
MODEL_CLASS = T2T
