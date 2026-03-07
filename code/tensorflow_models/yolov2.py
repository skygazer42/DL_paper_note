from __future__ import annotations

from .base import TensorFlowModelBase


class YOLOv2(TensorFlowModelBase):
    model_id = "yolov2"


MODEL_ID = "yolov2"
MODEL_CLASS = YOLOv2
