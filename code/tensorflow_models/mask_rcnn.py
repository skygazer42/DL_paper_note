from __future__ import annotations

from .base import TensorFlowModelBase


class MaskRCNN(TensorFlowModelBase):
    model_id = "mask_rcnn"


MODEL_ID = "mask_rcnn"
MODEL_CLASS = MaskRCNN
