from __future__ import annotations

from .base import PyTorchModelBase


class GAN(PyTorchModelBase):
    model_id = "gan"


MODEL_ID = "gan"
MODEL_CLASS = GAN
