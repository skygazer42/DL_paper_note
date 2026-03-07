from __future__ import annotations

from .base import PyTorchModelBase


class YOLOv4(PyTorchModelBase):
    model_id = "yolov4"


MODEL_ID = "yolov4"
MODEL_CLASS = YOLOv4
