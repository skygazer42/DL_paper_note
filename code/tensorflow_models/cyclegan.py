from __future__ import annotations

from .base import TensorFlowModelBase


class CycleGAN(TensorFlowModelBase):
    model_id = "cyclegan"


MODEL_ID = "cyclegan"
MODEL_CLASS = CycleGAN
