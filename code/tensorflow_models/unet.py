from __future__ import annotations

from .base import TensorFlowModelBase


class UNet(TensorFlowModelBase):
    model_id = "unet"


MODEL_ID = "unet"
MODEL_CLASS = UNet
