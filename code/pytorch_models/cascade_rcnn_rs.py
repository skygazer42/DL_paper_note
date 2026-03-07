from __future__ import annotations

from .base import PyTorchModelBase


class CascadeRCNNRS(PyTorchModelBase):
    model_id = "cascade_rcnn_rs"


MODEL_ID = "cascade_rcnn_rs"
MODEL_CLASS = CascadeRCNNRS
