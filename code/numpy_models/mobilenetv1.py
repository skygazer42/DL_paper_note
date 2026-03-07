from __future__ import annotations

from .base import NumpyModelBase


class MobileNetv1(NumpyModelBase):
    model_id = "mobilenetv1"


MODEL_ID = "mobilenetv1"
MODEL_CLASS = MobileNetv1
