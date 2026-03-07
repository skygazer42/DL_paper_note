from __future__ import annotations

from .base import NumpyModelBase


class ShuffleNetv1(NumpyModelBase):
    model_id = "shufflenetv1"


MODEL_ID = "shufflenetv1"
MODEL_CLASS = ShuffleNetv1
