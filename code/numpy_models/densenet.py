from __future__ import annotations

from .base import NumpyModelBase


class DenseNet(NumpyModelBase):
    model_id = "densenet"


MODEL_ID = "densenet"
MODEL_CLASS = DenseNet
