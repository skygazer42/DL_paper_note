from __future__ import annotations

import importlib
from pathlib import Path


def create(model_id: str):
    mod = importlib.import_module(f"tensorflow_models.{model_id}")
    return mod.MODEL_CLASS()


def available_model_ids() -> list[str]:
    pkg_dir = Path(__file__).resolve().parent
    return sorted([p.stem for p in pkg_dir.glob("*.py") if p.name != "__init__.py"])
