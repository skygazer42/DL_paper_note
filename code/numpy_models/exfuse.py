from __future__ import annotations

from .base import NumpyModelBase


class ExFuse(NumpyModelBase):
    model_id = "exfuse"


MODEL_ID = "exfuse"
MODEL_CLASS = ExFuse
