from __future__ import annotations

from .base import PyTorchModelBase


class Pix2pix(PyTorchModelBase):
    model_id = "pix2pix"


MODEL_ID = "pix2pix"
MODEL_CLASS = Pix2pix
