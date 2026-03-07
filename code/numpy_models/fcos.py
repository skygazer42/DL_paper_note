from __future__ import annotations

from .base import NumpyModelBase


class FCOS(NumpyModelBase):
    model_id = "fcos"


MODEL_ID = "fcos"
MODEL_CLASS = FCOS
