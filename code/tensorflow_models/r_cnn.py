from __future__ import annotations

from .base import TensorFlowModelBase


class RCNN(TensorFlowModelBase):
    model_id = "r_cnn"


MODEL_ID = "r_cnn"
MODEL_CLASS = RCNN
