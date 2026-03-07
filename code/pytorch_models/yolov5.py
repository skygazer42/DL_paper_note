from __future__ import annotations

from .base import PyTorchModelBase


class YOLOv5(PyTorchModelBase):
    model_id = "yolov5"


MODEL_ID = "yolov5"
MODEL_CLASS = YOLOv5
