from __future__ import annotations

from .base import NumpyModelBase


class ResNet(NumpyModelBase):
    model_id = "resnet"


MODEL_ID = "resnet"
MODEL_CLASS = ResNet
