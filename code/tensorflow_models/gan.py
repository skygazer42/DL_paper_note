from __future__ import annotations

from .base import TensorFlowModelBase


class GAN(TensorFlowModelBase):
    model_id = "gan"


MODEL_ID = "gan"
MODEL_CLASS = GAN
