from __future__ import annotations

from .base import PyTorchModelBase


class DeconvNet(PyTorchModelBase):
    model_id = "deconvnet"


MODEL_ID = "deconvnet"
MODEL_CLASS = DeconvNet
