from __future__ import annotations

from .base import NumpyModelBase


class DeepLabv1(NumpyModelBase):
    model_id = "deeplabv1"


MODEL_ID = "deeplabv1"
MODEL_CLASS = DeepLabv1
