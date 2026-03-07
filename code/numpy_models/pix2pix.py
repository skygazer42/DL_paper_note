from __future__ import annotations

from .base import NumpyModelBase


class Pix2pix(NumpyModelBase):
    model_id = "pix2pix"


MODEL_ID = "pix2pix"
MODEL_CLASS = Pix2pix
