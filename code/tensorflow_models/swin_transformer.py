from __future__ import annotations

from .base import TensorFlowModelBase


class SwinTransformer(TensorFlowModelBase):
    model_id = "swin_transformer"


MODEL_ID = "swin_transformer"
MODEL_CLASS = SwinTransformer
