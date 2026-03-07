from __future__ import annotations

from .base import PyTorchModelBase


class PPYOLOE(PyTorchModelBase):
    model_id = "ppyoloe"


MODEL_ID = "ppyoloe"
MODEL_CLASS = PPYOLOE
