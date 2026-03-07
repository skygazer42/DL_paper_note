from __future__ import annotations

from .base import NumpyModelBase


class YOLOv3(NumpyModelBase):
    model_id = "yolov3"


MODEL_ID = "yolov3"
MODEL_CLASS = YOLOv3
