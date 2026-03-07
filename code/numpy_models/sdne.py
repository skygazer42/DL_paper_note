from __future__ import annotations

from .base import NumpyModelBase


class SDNE(NumpyModelBase):
    model_id = "sdne"


MODEL_ID = "sdne"
MODEL_CLASS = SDNE
