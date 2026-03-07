from __future__ import annotations

from .base import NumpyModelBase


class InceptionV3(NumpyModelBase):
    model_id = "inceptionv3"


MODEL_ID = "inceptionv3"
MODEL_CLASS = InceptionV3
