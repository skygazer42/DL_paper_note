
from typing import Callable

from ..registry import MODEL_SPECS


def get_builder(model_id: str) -> Callable:
    # Imported lazily so backends can be imported without pulling in all model code.
    from .builders import BUILDERS

    if model_id not in BUILDERS:
        raise NotImplementedError(f"No builder registered for model_id={model_id!r}")
    return BUILDERS[model_id]


def all_model_ids() -> list[str]:
    return list(MODEL_SPECS.keys())

