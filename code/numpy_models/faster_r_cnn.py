from __future__ import annotations

from .base import NumpyModelBase


class FasterRCNN(NumpyModelBase):
    model_id = "faster_r_cnn"


MODEL_ID = "faster_r_cnn"
MODEL_CLASS = FasterRCNN
