from __future__ import annotations

from .base import NumpyModelBase


class YOLOv1(NumpyModelBase):
    model_id = "yolov1"


MODEL_ID = "yolov1"
MODEL_CLASS = YOLOv1
