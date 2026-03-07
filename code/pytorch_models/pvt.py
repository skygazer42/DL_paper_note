from __future__ import annotations

from .base import PyTorchModelBase


class PVT(PyTorchModelBase):
    model_id = "pvt"


MODEL_ID = "pvt"
MODEL_CLASS = PVT
