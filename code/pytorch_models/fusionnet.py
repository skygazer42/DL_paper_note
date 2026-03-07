from __future__ import annotations

from .base import PyTorchModelBase


class FusionNet(PyTorchModelBase):
    model_id = "fusionnet"


MODEL_ID = "fusionnet"
MODEL_CLASS = FusionNet
