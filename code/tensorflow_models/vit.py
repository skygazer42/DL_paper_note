from __future__ import annotations

from .base import TensorFlowModelBase


class ViT(TensorFlowModelBase):
    model_id = "vit"


MODEL_ID = "vit"
MODEL_CLASS = ViT
