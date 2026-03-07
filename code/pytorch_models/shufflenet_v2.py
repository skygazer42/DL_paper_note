from __future__ import annotations

from .base import PyTorchModelBase


class ShuffleNetV2(PyTorchModelBase):
    model_id = "shufflenet_v2"


MODEL_ID = "shufflenet_v2"
MODEL_CLASS = ShuffleNetV2
