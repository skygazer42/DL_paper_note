from __future__ import annotations

from .base import PyTorchModelBase


class RCNN(PyTorchModelBase):
    model_id = "r_cnn"


MODEL_ID = "r_cnn"
MODEL_CLASS = RCNN
