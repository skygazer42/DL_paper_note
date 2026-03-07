from __future__ import annotations

from .base import NumpyModelBase


class DeepLabv3Plus(NumpyModelBase):
    model_id = "deeplabv3_plus"


MODEL_ID = "deeplabv3_plus"
MODEL_CLASS = DeepLabv3Plus
