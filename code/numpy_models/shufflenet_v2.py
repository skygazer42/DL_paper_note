from __future__ import annotations

from .base import NumpyModelBase


class ShuffleNetV2(NumpyModelBase):
    model_id = "shufflenet_v2"


MODEL_ID = "shufflenet_v2"
MODEL_CLASS = ShuffleNetV2
