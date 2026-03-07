from __future__ import annotations

from .base import NumpyModelBase


class Deit(NumpyModelBase):
    model_id = "deit"


MODEL_ID = "deit"
MODEL_CLASS = Deit
