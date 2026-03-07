from __future__ import annotations

from .base import NumpyModelBase


class FastRCNN(NumpyModelBase):
    model_id = "fast_r_cnn"


MODEL_ID = "fast_r_cnn"
MODEL_CLASS = FastRCNN
