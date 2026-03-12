
import argparse
import importlib
import sys
from typing import Any

import numpy as np

from cv_models.registry import MODEL_SPECS
from cv_models.sample_inputs import make_sample_inputs


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    # torch.Tensor
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    # tf.Tensor
    if hasattr(x, "numpy"):
        return x.numpy()
    raise TypeError(f"Unsupported tensor type: {type(x)!r}")


def _run_one(backend: str, model_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    backend_pkg = {
        "numpy": "numpy_models",
        "torch": "pytorch_models",
        "tf": "tensorflow_models",
    }.get(backend)
    if backend_pkg is None:
        raise KeyError(f"Unknown backend: {backend!r}")
    mod = importlib.import_module(f"{backend_pkg}.{model_id}")
    model = mod.MODEL_CLASS()
    return model.forward(inputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare forward outputs across backends for the same model.")
    parser.add_argument("--model", default="resnet", help="model_id (e.g. resnet, faster_r_cnn, unet)")
    parser.add_argument(
        "--backends",
        default="numpy,torch",
        help="Comma-separated list: numpy,torch,tf (requires corresponding libs installed)",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)

    model_id = args.model.strip()
    if model_id not in MODEL_SPECS:
        raise SystemExit(f"Unknown model_id: {model_id!r}. Use smoke_test --list to see valid ids.")

    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    if len(backends) < 2:
        raise SystemExit("--backends must include at least two backends")

    inputs = make_sample_inputs(model_id, seed=0)
    results: dict[str, dict[str, Any]] = {}
    for b in backends:
        try:
            results[b] = _run_one(b, model_id, inputs)
        except Exception as e:
            print(f"FAILED backend={b}: {e}", file=sys.stderr)
            return 2

    # Compare everything to the first backend as reference.
    ref_backend = backends[0]
    ref = results[ref_backend]
    ok = True

    for b in backends[1:]:
        cur = results[b]
        keys = sorted(set(ref.keys()) | set(cur.keys()))
        for k in keys:
            if k not in ref:
                print(f"[{b}] extra key: {k!r}")
                ok = False
                continue
            if k not in cur:
                print(f"[{b}] missing key: {k!r}")
                ok = False
                continue
            a = _to_numpy(ref[k]).astype(np.float32, copy=False)
            c = _to_numpy(cur[k]).astype(np.float32, copy=False)
            if a.shape != c.shape:
                print(f"[{b}] shape mismatch for {k!r}: {a.shape} vs {c.shape}")
                ok = False
                continue
            diff = np.max(np.abs(a - c)) if a.size else 0.0
            close = np.allclose(a, c, atol=args.atol, rtol=args.rtol)
            print(f"[{ref_backend} vs {b}] {k}: shape={a.shape} max_abs_diff={diff:.6g} allclose={close}")
            ok = ok and close

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

