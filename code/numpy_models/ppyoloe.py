from __future__ import annotations

from .base import NumpyModelBase


class PPYOLOE(NumpyModelBase):
    model_id = "ppyoloe"


MODEL_ID = "ppyoloe"
MODEL_CLASS = PPYOLOE
