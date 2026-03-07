from __future__ import annotations

from .base import TensorFlowModelBase


class VGG(TensorFlowModelBase):
    model_id = "vgg"


MODEL_ID = "vgg"
MODEL_CLASS = VGG
