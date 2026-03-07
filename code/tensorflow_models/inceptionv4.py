from __future__ import annotations

from .base import TensorFlowModelBase


class InceptionV4(TensorFlowModelBase):
    model_id = "inceptionv4"


MODEL_ID = "inceptionv4"
MODEL_CLASS = InceptionV4
