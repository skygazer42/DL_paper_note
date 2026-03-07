from __future__ import annotations

from .base import TensorFlowModelBase


class YOLOv1(TensorFlowModelBase):
    model_id = "yolov1"


MODEL_ID = "yolov1"
MODEL_CLASS = YOLOv1
