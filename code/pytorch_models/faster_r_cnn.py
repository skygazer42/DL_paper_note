from __future__ import annotations

from .base import PyTorchModelBase


class FasterRCNN(PyTorchModelBase):
    model_id = "faster_r_cnn"


MODEL_ID = "faster_r_cnn"
MODEL_CLASS = FasterRCNN
