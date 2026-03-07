from __future__ import annotations

from .base import TensorFlowModelBase


class DenseNet(TensorFlowModelBase):
    model_id = "densenet"


MODEL_ID = "densenet"
MODEL_CLASS = DenseNet
