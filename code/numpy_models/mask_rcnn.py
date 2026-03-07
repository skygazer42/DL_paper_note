from __future__ import annotations

from .base import NumpyModelBase


class MaskRCNN(NumpyModelBase):
    model_id = "mask_rcnn"


MODEL_ID = "mask_rcnn"
MODEL_CLASS = MaskRCNN
