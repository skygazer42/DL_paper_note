from __future__ import annotations

from .base import PyTorchModelBase


class InceptionV3(PyTorchModelBase):
    model_id = "inceptionv3"


MODEL_ID = "inceptionv3"
MODEL_CLASS = InceptionV3
