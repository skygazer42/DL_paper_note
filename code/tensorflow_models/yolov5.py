from __future__ import annotations

from .base import TensorFlowModelBase


class YOLOv5(TensorFlowModelBase):
    model_id = "yolov5"


MODEL_ID = "yolov5"
MODEL_CLASS = YOLOv5
