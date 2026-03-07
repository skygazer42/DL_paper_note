from __future__ import annotations

from .base import NumpyModelBase


class RetinaNet(NumpyModelBase):
    model_id = "retinanet"


MODEL_ID = "retinanet"
MODEL_CLASS = RetinaNet
