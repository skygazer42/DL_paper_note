from __future__ import annotations

from .base import PyTorchModelBase


class Deit(PyTorchModelBase):
    model_id = "deit"


MODEL_ID = "deit"
MODEL_CLASS = Deit
