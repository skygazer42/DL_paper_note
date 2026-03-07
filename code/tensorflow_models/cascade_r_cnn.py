from __future__ import annotations

from .base import TensorFlowModelBase


class CascadeRCNN(TensorFlowModelBase):
    model_id = "cascade_r_cnn"


MODEL_ID = "cascade_r_cnn"
MODEL_CLASS = CascadeRCNN
