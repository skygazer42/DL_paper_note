from __future__ import annotations

from .base import PyTorchModelBase


class YOLOv2(PyTorchModelBase):
    model_id = "yolov2"


MODEL_ID = "yolov2"
MODEL_CLASS = YOLOv2
