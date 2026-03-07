from __future__ import annotations

from .base import TensorFlowModelBase


class BiSeNetv1(TensorFlowModelBase):
    model_id = "bisenetv1"


MODEL_ID = "bisenetv1"
MODEL_CLASS = BiSeNetv1
