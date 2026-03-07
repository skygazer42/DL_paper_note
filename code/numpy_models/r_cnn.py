from __future__ import annotations

from .base import NumpyModelBase


class RCNN(NumpyModelBase):
    model_id = "r_cnn"


MODEL_ID = "r_cnn"
MODEL_CLASS = RCNN
