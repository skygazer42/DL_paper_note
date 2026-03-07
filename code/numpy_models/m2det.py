from __future__ import annotations

from .base import NumpyModelBase


class M2Det(NumpyModelBase):
    model_id = "m2det"


MODEL_ID = "m2det"
MODEL_CLASS = M2Det
