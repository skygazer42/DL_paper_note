from __future__ import annotations

from .base import TensorFlowModelBase


class YOLOv3(TensorFlowModelBase):
    model_id = "yolov3"


MODEL_ID = "yolov3"
MODEL_CLASS = YOLOv3
