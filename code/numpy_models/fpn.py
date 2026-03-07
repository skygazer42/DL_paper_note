from __future__ import annotations

from .base import NumpyModelBase


class FPN(NumpyModelBase):
    model_id = "fpn"


MODEL_ID = "fpn"
MODEL_CLASS = FPN
