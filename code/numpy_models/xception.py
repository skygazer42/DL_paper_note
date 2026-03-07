from __future__ import annotations

from .base import NumpyModelBase


class Xception(NumpyModelBase):
    model_id = "xception"


MODEL_ID = "xception"
MODEL_CLASS = Xception
