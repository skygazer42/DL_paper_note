from __future__ import annotations

from .base import TensorFlowModelBase


class YOLOv4(TensorFlowModelBase):
    model_id = "yolov4"


MODEL_ID = "yolov4"
MODEL_CLASS = YOLOv4
