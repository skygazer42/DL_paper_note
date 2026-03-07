from __future__ import annotations

from .base import NumpyModelBase


class FCN(NumpyModelBase):
    model_id = "fcn"


MODEL_ID = "fcn"
MODEL_CLASS = FCN
