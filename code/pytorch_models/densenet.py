from __future__ import annotations

from .base import PyTorchModelBase


class DenseNet(PyTorchModelBase):
    model_id = "densenet"


MODEL_ID = "densenet"
MODEL_CLASS = DenseNet
