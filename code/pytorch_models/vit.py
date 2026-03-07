from __future__ import annotations

from .base import PyTorchModelBase


class ViT(PyTorchModelBase):
    model_id = "vit"


MODEL_ID = "vit"
MODEL_CLASS = ViT
