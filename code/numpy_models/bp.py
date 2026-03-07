from __future__ import annotations

from .base import NumpyModelBase


class BP(NumpyModelBase):
    model_id = "bp"


MODEL_ID = "bp"
MODEL_CLASS = BP
