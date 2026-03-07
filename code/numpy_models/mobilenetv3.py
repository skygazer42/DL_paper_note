from __future__ import annotations

from .base import NumpyModelBase


class MobileNetV3(NumpyModelBase):
    model_id = "mobilenetv3"


MODEL_ID = "mobilenetv3"
MODEL_CLASS = MobileNetV3
