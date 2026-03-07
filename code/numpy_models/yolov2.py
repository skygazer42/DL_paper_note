from __future__ import annotations

from .base import NumpyModelBase


class YOLOv2(NumpyModelBase):
    model_id = "yolov2"


MODEL_ID = "yolov2"
MODEL_CLASS = YOLOv2
