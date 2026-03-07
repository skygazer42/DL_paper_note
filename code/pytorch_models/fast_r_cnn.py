from __future__ import annotations

from .base import PyTorchModelBase


class FastRCNN(PyTorchModelBase):
    model_id = "fast_r_cnn"


MODEL_ID = "fast_r_cnn"
MODEL_CLASS = FastRCNN
