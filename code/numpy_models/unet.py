from __future__ import annotations

from .base import NumpyModelBase


class UNet(NumpyModelBase):
    model_id = "unet"


MODEL_ID = "unet"
MODEL_CLASS = UNet
