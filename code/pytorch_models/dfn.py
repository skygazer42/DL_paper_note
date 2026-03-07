from __future__ import annotations

from .base import PyTorchModelBase


class DFN(PyTorchModelBase):
    model_id = "dfn"


MODEL_ID = "dfn"
MODEL_CLASS = DFN
