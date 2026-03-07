from __future__ import annotations

from .base import PyTorchModelBase


class FCOS(PyTorchModelBase):
    model_id = "fcos"


MODEL_ID = "fcos"
MODEL_CLASS = FCOS
