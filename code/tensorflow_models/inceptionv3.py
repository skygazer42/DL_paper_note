from __future__ import annotations

from .base import TensorFlowModelBase


class InceptionV3(TensorFlowModelBase):
    model_id = "inceptionv3"


MODEL_ID = "inceptionv3"
MODEL_CLASS = InceptionV3
