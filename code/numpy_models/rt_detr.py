from __future__ import annotations

from .base import NumpyModelBase


class RTDETR(NumpyModelBase):
    model_id = "rt_detr"


MODEL_ID = "rt_detr"
MODEL_CLASS = RTDETR
