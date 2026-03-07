from __future__ import annotations

from .base import PyTorchModelBase


class ZFNet(PyTorchModelBase):
    model_id = "zfnet"


MODEL_ID = "zfnet"
MODEL_CLASS = ZFNet
