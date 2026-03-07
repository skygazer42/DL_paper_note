from __future__ import annotations

from .base import PyTorchModelBase


class SwinTransformer(PyTorchModelBase):
    model_id = "swin_transformer"


MODEL_ID = "swin_transformer"
MODEL_CLASS = SwinTransformer
