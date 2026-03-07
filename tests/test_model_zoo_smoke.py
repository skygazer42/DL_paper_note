import re
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _readme_model_names() -> list[str]:
    readme_path = _repo_root() / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    names: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*-\s*\*\*(.+?)\*\*", line)
        if not m:
            continue
        raw = m.group(1).strip()

        # Cleanup for a couple of bullets that include extra markdown.
        raw = raw.rstrip(":").strip()
        if ":" in raw:
            raw = raw.split(":", 1)[0].strip()
        names.append(raw)

    # Keep README order stable.
    return names


def _import_cv_models_or_fail():
    # Make `code/` importable (this repo intentionally stores implementation under `code/`).
    code_dir = _repo_root() / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    try:
        import cv_models  # noqa: F401
        from cv_models import registry  # noqa: F401
        from cv_models.build import build_model  # noqa: F401
        from cv_models.sample_inputs import make_sample_inputs  # noqa: F401
    except Exception as e:  # pragma: no cover - this is the expected initial RED failure
        pytest.fail(f"cv_models not implemented yet (import failed): {e}")


def test_registry_covers_all_readme_models():
    _import_cv_models_or_fail()

    from cv_models import registry

    expected = _readme_model_names()
    assert len(expected) >= 50, "README parsing looks wrong; expected many model bullets"
    assert len(expected) == len(set(expected)), "README model bullet names must be unique"

    assert hasattr(registry, "README_MODEL_NAMES"), "registry.README_MODEL_NAMES missing"
    assert registry.README_MODEL_NAMES == expected

    assert hasattr(registry, "MODEL_SPECS"), "registry.MODEL_SPECS missing"
    assert len(registry.MODEL_SPECS) == len(expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_build_and_forward_all_models_numpy_and_torch(backend: str):
    _import_cv_models_or_fail()

    from cv_models import registry
    from cv_models.build import build_model
    from cv_models.sample_inputs import make_sample_inputs

    for spec in registry.MODEL_SPECS.values():
        model = build_model(spec.model_id, backend=backend)
        inputs = make_sample_inputs(spec.model_id)
        outputs = model(inputs)

        assert isinstance(outputs, dict), f"{spec.model_id} must return a dict"
        assert outputs, f"{spec.model_id} returned empty outputs"

        # Each output should be a tensor/array-like with a non-empty shape.
        for key, value in outputs.items():
            assert hasattr(value, "shape"), f"{spec.model_id}.{key} must have .shape"
            assert all(int(d) >= 0 for d in value.shape), f"{spec.model_id}.{key} invalid shape"

        # Lightweight backend-specific type checks
        if backend == "numpy":
            import numpy as np

            assert all(
                isinstance(v, np.ndarray) for v in outputs.values()
            ), f"{spec.model_id} numpy backend must return np.ndarray values"
        elif backend == "torch":
            import torch

            assert all(
                isinstance(v, torch.Tensor) for v in outputs.values()
            ), f"{spec.model_id} torch backend must return torch.Tensor values"


def test_tensorflow_backend_optional_smoke():
    _import_cv_models_or_fail()

    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        pytest.skip("tensorflow not installed in this environment")

    from cv_models import registry
    from cv_models.build import build_model
    from cv_models.sample_inputs import make_sample_inputs

    # Just a small representative subset so CI doesn't get too slow.
    subset = ["vgg", "resnet", "unet", "vit"]
    for model_id in subset:
        assert model_id in registry.MODEL_SPECS
        model = build_model(model_id, backend="tf")
        outputs = model(make_sample_inputs(model_id))
        assert isinstance(outputs, dict)
        assert outputs

