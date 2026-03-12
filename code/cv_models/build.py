
from typing import Literal

from .registry import MODEL_SPECS

BackendName = Literal["numpy", "torch", "tf"]


def build_model(model_id: str, *, backend: BackendName):
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Unknown model_id: {model_id!r}")

    from .backends import get_backend
    from .models import get_builder

    backend_impl = get_backend(backend)
    builder = get_builder(model_id)
    return builder(backend_impl)

