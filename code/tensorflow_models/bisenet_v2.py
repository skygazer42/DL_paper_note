from __future__ import annotations

from .base import TensorFlowModelBase


class BiSeNetV2(TensorFlowModelBase):
    model_id = "bisenet_v2"


MODEL_ID = "bisenet_v2"
MODEL_CLASS = BiSeNetV2
