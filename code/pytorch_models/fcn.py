from __future__ import annotations

from .base import PyTorchModelBase


class FCN(PyTorchModelBase):
    model_id = "fcn"


MODEL_ID = "fcn"
MODEL_CLASS = FCN
