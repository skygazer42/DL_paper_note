from __future__ import annotations

from .base import NumpyModelBase


class EfficientDet(NumpyModelBase):
    model_id = "efficientdet"


MODEL_ID = "efficientdet"
MODEL_CLASS = EfficientDet
