from __future__ import annotations

from .base import TensorFlowModelBase


class DeepLabv3(TensorFlowModelBase):
    model_id = "deeplabv3"


MODEL_ID = "deeplabv3"
MODEL_CLASS = DeepLabv3
