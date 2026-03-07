from __future__ import annotations

import importlib


def create(model_id: str):
    mod = importlib.import_module(f"pytorch_models.{model_id}")
    return mod.MODEL_CLASS()


def available_model_ids() -> list[str]:
    from cv_models.registry import MODEL_SPECS

    return list(MODEL_SPECS.keys())
