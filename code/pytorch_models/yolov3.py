from __future__ import annotations

from .base import PyTorchModelBase


class YOLOv3(PyTorchModelBase):
    model_id = "yolov3"


MODEL_ID = "yolov3"
MODEL_CLASS = YOLOv3
