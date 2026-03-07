from __future__ import annotations

from .base import NumpyModelBase


class SegNet(NumpyModelBase):
    model_id = "segnet"


MODEL_ID = "segnet"
MODEL_CLASS = SegNet
