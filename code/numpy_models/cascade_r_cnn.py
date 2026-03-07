from __future__ import annotations

from .base import NumpyModelBase


class CascadeRCNN(NumpyModelBase):
    model_id = "cascade_r_cnn"


MODEL_ID = "cascade_r_cnn"
MODEL_CLASS = CascadeRCNN
