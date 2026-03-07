from __future__ import annotations

from .base import NumpyModelBase


class MAE(NumpyModelBase):
    model_id = "mae"


MODEL_ID = "mae"
MODEL_CLASS = MAE
