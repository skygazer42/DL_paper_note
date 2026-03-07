from __future__ import annotations

from .base import PyTorchModelBase


class MaskRCNN(PyTorchModelBase):
    model_id = "mask_rcnn"


MODEL_ID = "mask_rcnn"
MODEL_CLASS = MaskRCNN
