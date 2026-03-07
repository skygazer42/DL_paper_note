from __future__ import annotations

from .base import TensorFlowModelBase


class DeepLabv3Plus(TensorFlowModelBase):
    model_id = "deeplabv3_plus"


MODEL_ID = "deeplabv3_plus"
MODEL_CLASS = DeepLabv3Plus
