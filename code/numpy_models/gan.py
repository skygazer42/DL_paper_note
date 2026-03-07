from __future__ import annotations

from .base import NumpyModelBase


class GAN(NumpyModelBase):
    model_id = "gan"


MODEL_ID = "gan"
MODEL_CLASS = GAN
