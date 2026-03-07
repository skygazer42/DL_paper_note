from __future__ import annotations

from .base import PyTorchModelBase


class CycleGAN(PyTorchModelBase):
    model_id = "cyclegan"


MODEL_ID = "cyclegan"
MODEL_CLASS = CycleGAN
