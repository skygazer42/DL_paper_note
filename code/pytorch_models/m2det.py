from __future__ import annotations

from .base import PyTorchModelBase


class M2Det(PyTorchModelBase):
    model_id = "m2det"


MODEL_ID = "m2det"
MODEL_CLASS = M2Det
