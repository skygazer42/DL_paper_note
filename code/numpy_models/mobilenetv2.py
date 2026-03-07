from __future__ import annotations

from .base import NumpyModelBase


class MobileNetV2(NumpyModelBase):
    model_id = "mobilenetv2"


MODEL_ID = "mobilenetv2"
MODEL_CLASS = MobileNetV2
