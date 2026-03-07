from __future__ import annotations

from .base import NumpyModelBase


class T2T(NumpyModelBase):
    model_id = "t2t"


MODEL_ID = "t2t"
MODEL_CLASS = T2T
