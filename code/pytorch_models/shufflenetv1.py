from __future__ import annotations

from .base import PyTorchModelBase


class ShuffleNetv1(PyTorchModelBase):
    model_id = "shufflenetv1"


MODEL_ID = "shufflenetv1"
MODEL_CLASS = ShuffleNetv1
