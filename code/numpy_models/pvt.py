from __future__ import annotations

from .base import NumpyModelBase


class PVT(NumpyModelBase):
    model_id = "pvt"


MODEL_ID = "pvt"
MODEL_CLASS = PVT
