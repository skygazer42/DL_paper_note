from __future__ import annotations

from .base import PyTorchModelBase


class UNet(PyTorchModelBase):
    model_id = "unet"


MODEL_ID = "unet"
MODEL_CLASS = UNet
