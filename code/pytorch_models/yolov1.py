from __future__ import annotations

from .base import PyTorchModelBase


class YOLOv1(PyTorchModelBase):
    model_id = "yolov1"


MODEL_ID = "yolov1"
MODEL_CLASS = YOLOv1
