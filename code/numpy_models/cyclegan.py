from __future__ import annotations

from .base import NumpyModelBase


class CycleGAN(NumpyModelBase):
    model_id = "cyclegan"


MODEL_ID = "cyclegan"
MODEL_CLASS = CycleGAN
