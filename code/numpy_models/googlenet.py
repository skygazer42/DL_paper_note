from __future__ import annotations

from .base import NumpyModelBase


class GoogleNet(NumpyModelBase):
    model_id = "googlenet"


MODEL_ID = "googlenet"
MODEL_CLASS = GoogleNet
