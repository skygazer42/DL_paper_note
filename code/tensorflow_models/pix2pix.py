from __future__ import annotations

from .base import TensorFlowModelBase


class Pix2pix(TensorFlowModelBase):
    model_id = "pix2pix"


MODEL_ID = "pix2pix"
MODEL_CLASS = Pix2pix
