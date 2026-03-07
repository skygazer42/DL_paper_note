from __future__ import annotations

from .base import NumpyModelBase


class DeepLabv3(NumpyModelBase):
    model_id = "deeplabv3"


MODEL_ID = "deeplabv3"
MODEL_CLASS = DeepLabv3
