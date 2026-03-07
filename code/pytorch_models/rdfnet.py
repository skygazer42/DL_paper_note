from __future__ import annotations

from .base import PyTorchModelBase


class RDFNet(PyTorchModelBase):
    model_id = "rdfnet"


MODEL_ID = "rdfnet"
MODEL_CLASS = RDFNet
