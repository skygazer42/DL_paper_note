from __future__ import annotations

from .base import NumpyModelBase


class VGG(NumpyModelBase):
    model_id = "vgg"


MODEL_ID = "vgg"
MODEL_CLASS = VGG
