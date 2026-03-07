from __future__ import annotations

from .base import NumpyModelBase


class SeNet(NumpyModelBase):
    model_id = "senet"


MODEL_ID = "senet"
MODEL_CLASS = SeNet
