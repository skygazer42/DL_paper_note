from __future__ import annotations

from .base import PyTorchModelBase


class CascadeRCNN(PyTorchModelBase):
    model_id = "cascade_r_cnn"


MODEL_ID = "cascade_r_cnn"
MODEL_CLASS = CascadeRCNN
