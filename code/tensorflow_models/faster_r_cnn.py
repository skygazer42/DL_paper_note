from __future__ import annotations

from .base import TensorFlowModelBase


class FasterRCNN(TensorFlowModelBase):
    model_id = "faster_r_cnn"


MODEL_ID = "faster_r_cnn"
MODEL_CLASS = FasterRCNN
