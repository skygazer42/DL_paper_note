from __future__ import annotations

from .base import NumpyModelBase


class InceptionV4(NumpyModelBase):
    model_id = "inceptionv4"


MODEL_ID = "inceptionv4"
MODEL_CLASS = InceptionV4
