from __future__ import annotations

import numpy as np

from .registry import MODEL_SPECS


def make_sample_inputs(model_id: str, *, seed: int = 0) -> dict:
    """
    Generate small random inputs for smoke testing.

    All inputs are returned as NumPy arrays; backend conversion happens inside the model.
    """
    rng = np.random.default_rng(seed)
    spec = MODEL_SPECS[model_id]

    if spec.task in {"classification", "detection", "segmentation", "transformer"}:
        # NHWC image
        image = rng.standard_normal((1, 32, 32, 3), dtype=np.float32)
        return {"image": image}

    if spec.task == "mlp":
        x = rng.standard_normal((2, 128), dtype=np.float32)
        return {"x": x}

    if spec.task == "gan":
        if spec.readme_name in {"pix2pix", "CycleGAN"}:
            image = rng.standard_normal((1, 32, 32, 3), dtype=np.float32)
            return {"image": image}
        z = rng.standard_normal((2, 64), dtype=np.float32)
        return {"z": z}

    if spec.task == "graph_pair":
        num_nodes = 8
        src = np.array([0, 1, 2, 3], dtype=np.int64)
        dst = np.array([1, 2, 3, 4], dtype=np.int64)
        return {"num_nodes": num_nodes, "src": src, "dst": dst}

    if spec.task == "graph_adj":
        num_nodes = 8
        x = rng.standard_normal((num_nodes, 16), dtype=np.float32)
        adj = (rng.random((num_nodes, num_nodes)) > 0.75).astype(np.float32)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 1.0)
        return {"x": x, "adj": adj}

    raise ValueError(f"Unknown task for model_id={model_id!r}: {spec.task!r}")

