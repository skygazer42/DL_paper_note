from __future__ import annotations

from .base import NumpyModelBase


class EfficientNet(NumpyModelBase):
    model_id = "efficientnet"


MODEL_ID = "efficientnet"
MODEL_CLASS = EfficientNet
