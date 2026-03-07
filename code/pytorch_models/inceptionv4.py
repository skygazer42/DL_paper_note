from __future__ import annotations

from .base import PyTorchModelBase


class InceptionV4(PyTorchModelBase):
    model_id = "inceptionv4"


MODEL_ID = "inceptionv4"
MODEL_CLASS = InceptionV4
