from __future__ import annotations

from .base import NumpyModelBase


class SwinTransformer(NumpyModelBase):
    model_id = "swin_transformer"


MODEL_ID = "swin_transformer"
MODEL_CLASS = SwinTransformer
