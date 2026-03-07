from __future__ import annotations

from .base import PyTorchModelBase


class GCN(PyTorchModelBase):
    model_id = "gcn"


MODEL_ID = "gcn"
MODEL_CLASS = GCN
