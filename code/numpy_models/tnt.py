from __future__ import annotations

from .base import NumpyModelBase


class TnT(NumpyModelBase):
    model_id = "tnt"


MODEL_ID = "tnt"
MODEL_CLASS = TnT
