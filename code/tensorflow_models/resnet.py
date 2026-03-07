from __future__ import annotations

from .base import TensorFlowModelBase


class ResNet(TensorFlowModelBase):
    model_id = "resnet"


MODEL_ID = "resnet"
MODEL_CLASS = ResNet
