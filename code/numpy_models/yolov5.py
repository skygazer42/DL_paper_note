from __future__ import annotations

from .base import NumpyModelBase


class YOLOv5(NumpyModelBase):
    model_id = "yolov5"


MODEL_ID = "yolov5"
MODEL_CLASS = YOLOv5
