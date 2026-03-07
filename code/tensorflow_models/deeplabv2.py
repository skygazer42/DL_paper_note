from __future__ import annotations

from .base import TensorFlowModelBase


class DeepLabv2(TensorFlowModelBase):
    model_id = "deeplabv2"


MODEL_ID = "deeplabv2"
MODEL_CLASS = DeepLabv2
