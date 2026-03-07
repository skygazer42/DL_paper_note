from __future__ import annotations

from .base import TensorFlowModelBase


class CascadeRCNNRS(TensorFlowModelBase):
    model_id = "cascade_rcnn_rs"


MODEL_ID = "cascade_rcnn_rs"
MODEL_CLASS = CascadeRCNNRS
