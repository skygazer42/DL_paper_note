from __future__ import annotations

from .base import NumpyModelBase


class ViT(NumpyModelBase):
    model_id = "vit"


MODEL_ID = "vit"
MODEL_CLASS = ViT
