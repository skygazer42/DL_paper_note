from __future__ import annotations

from .base import TensorFlowModelBase


class DeepLabv1(TensorFlowModelBase):
    model_id = "deeplabv1"


MODEL_ID = "deeplabv1"
MODEL_CLASS = DeepLabv1
