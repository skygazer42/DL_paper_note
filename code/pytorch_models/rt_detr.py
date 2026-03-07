from __future__ import annotations

from .base import PyTorchModelBase


class RTDETR(PyTorchModelBase):
    model_id = "rt_detr"


MODEL_ID = "rt_detr"
MODEL_CLASS = RTDETR
