from __future__ import annotations

from .base import PyTorchModelBase


class MAE(PyTorchModelBase):
    model_id = "mae"


MODEL_ID = "mae"
MODEL_CLASS = MAE
