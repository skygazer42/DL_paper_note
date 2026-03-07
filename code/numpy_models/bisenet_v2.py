from __future__ import annotations

from .base import NumpyModelBase


class BiSeNetV2(NumpyModelBase):
    model_id = "bisenet_v2"


MODEL_ID = "bisenet_v2"
MODEL_CLASS = BiSeNetV2
