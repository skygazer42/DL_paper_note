from __future__ import annotations

from .base import TensorFlowModelBase


class ShuffleNetv1(TensorFlowModelBase):
    model_id = "shufflenetv1"


MODEL_ID = "shufflenetv1"
MODEL_CLASS = ShuffleNetv1
