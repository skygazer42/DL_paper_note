from __future__ import annotations

import argparse
import importlib
import sys
from typing import Iterable

from cv_models.registry import MODEL_SPECS
from cv_models.sample_inputs import make_sample_inputs


def _iter_model_ids(arg: str) -> Iterable[str]:
    if arg == "all":
        return list(MODEL_SPECS.keys())
    if arg not in MODEL_SPECS:
        raise SystemExit(f"Unknown model_id: {arg!r}. Use --list to see valid ids.")
    return [arg]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test all forward-only toy models.")
    parser.add_argument("--backend", default="numpy", help="numpy | torch | tf (alias: pytorch -> torch)")
    parser.add_argument("--model", default="all", help="all | <model_id>")
    parser.add_argument("--list", action="store_true", help="List all model ids and exit")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failure")
    args = parser.parse_args(argv)

    if args.list:
        for mid, spec in MODEL_SPECS.items():
            print(f"{mid}\t({spec.readme_name})")
        return 0

    backend = args.backend.strip().lower()
    if backend == "pytorch":
        backend = "torch"

    backend_pkg = {
        "numpy": "numpy_models",
        "torch": "pytorch_models",
        "tf": "tensorflow_models",
    }.get(backend)
    if backend_pkg is None:
        raise SystemExit(f"Unknown backend: {backend!r}")

    model_ids = list(_iter_model_ids(args.model.strip()))
    failures: list[str] = []

    for model_id in model_ids:
        spec = MODEL_SPECS[model_id]
        try:
            mod = importlib.import_module(f"{backend_pkg}.{model_id}")
            model = mod.MODEL_CLASS()
            outputs = model.forward(make_sample_inputs(model_id))
            if not isinstance(outputs, dict) or not outputs:
                raise AssertionError("model must return a non-empty dict")
            for k, v in outputs.items():
                if not hasattr(v, "shape"):
                    raise AssertionError(f"output {k!r} missing .shape")
        except Exception as e:
            failures.append(f"{model_id} ({spec.readme_name}): {e}")
            if args.fail_fast:
                break

    if failures:
        print("SMOKE TEST FAILURES:", file=sys.stderr)
        for f in failures:
            print(f" - {f}", file=sys.stderr)
        return 1

    print(f"OK: {len(model_ids)} model(s) passed on backend={backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
