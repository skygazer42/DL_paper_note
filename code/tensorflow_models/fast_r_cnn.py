from __future__ import annotations

from .base import TensorFlowModelBase


class FastRCNN(TensorFlowModelBase):
    model_id = "fast_r_cnn"


MODEL_ID = "fast_r_cnn"
MODEL_CLASS = FastRCNN
