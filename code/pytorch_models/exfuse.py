from __future__ import annotations

from .base import PyTorchModelBase


class ExFuse(PyTorchModelBase):
    model_id = "exfuse"


MODEL_ID = "exfuse"
MODEL_CLASS = ExFuse
