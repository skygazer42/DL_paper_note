from __future__ import annotations

from .base import NumpyModelBase


class Transformer(NumpyModelBase):
    model_id = "transformer"


MODEL_ID = "transformer"
MODEL_CLASS = Transformer
