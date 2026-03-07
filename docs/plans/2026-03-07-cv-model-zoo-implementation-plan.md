# CV Paper Model Zoo (NumPy/TF/PyTorch) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `code/` folder that provides forward-only (“A 档”) reference implementations for every model name listed in `README.md`, with backends for **NumPy (only numpy)**, **TensorFlow**, and **PyTorch**, plus a smoke test runner.

**Architecture:** Implement a small backend-agnostic layer/ops surface (conv/linear/pool/attention/etc.) and define each paper model as a *toy but distinctive* network that captures the key idea (residual, dense concat, SE, depthwise, attention, etc.). Detection/segmentation models output **raw heads/logits** only (no training, no NMS/postprocess).

**Tech Stack:** Python 3.10, `numpy`, optional `torch`, optional `tensorflow`, `pytest`.

---

### Task 1: Add failing smoke tests (RED)

**Files:**
- Create: `tests/test_model_zoo_smoke.py`

**Step 1: Write the failing test**
- Parse model names from `README.md` (the `**NAME**` bullet format).
- Attempt to import the future registry `cv_models.registry` (from `code/` via `sys.path`).
- Assert:
  - Registry exists and covers all README models (same count).
  - Building each model for `numpy` and `torch` produces a callable.
  - Forward pass returns a dict of non-empty tensors/arrays.
  - TensorFlow backend is optional (skip if TF not installed).

**Step 2: Run test to verify it fails**

Run: `pytest -q`

Expected: FAIL (not error) with a clear message like “cv_models not implemented yet”.

---

### Task 2: Implement minimal scaffolding (GREEN)

**Files:**
- Create: `code/cv_models/__init__.py`
- Create: `code/cv_models/registry.py`
- Create: `code/cv_models/sample_inputs.py`
- Create: `code/cv_models/build.py`

**Steps:**
1. Implement a minimal `list_readme_models()` helper and a placeholder registry mapping.
2. Implement `build_model(model_id, backend=...)` that raises a clear `NotImplementedError` for missing models (so tests can move from “missing package” to “missing model”).
3. Add `sample_inputs(model_id)` to generate small random inputs for each task type.

**Verify:** `pytest -q` should move closer to green (still failing for missing implementations).

---

### Task 3: Implement backend ops surface

**Files:**
- Create: `code/cv_models/backends/__init__.py`
- Create: `code/cv_models/backends/numpy_ops.py`
- Create: `code/cv_models/backends/torch_ops.py`
- Create: `code/cv_models/backends/tf_ops.py`

**Scope (forward-only):**
- `conv2d`, grouped/depthwise conv, `linear`
- `relu`, `gelu`, `sigmoid`, `softmax`
- `max_pool2d`, `avg_pool2d`, `global_avg_pool2d`
- `upsample2d_nearest`, `concat`, `reshape`, `transpose`
- `layer_norm`
- Small attention helper for vision transformers

**Verify:** Run `pytest -q` and (optionally) run the smoke runner once implemented.

---

### Task 4: Implement model builders (cover all README names)

**Files:**
- Create: `code/cv_models/models/__init__.py`
- Create: `code/cv_models/models/classification.py`
- Create: `code/cv_models/models/detection.py`
- Create: `code/cv_models/models/segmentation.py`
- Create: `code/cv_models/models/transformers.py`
- Create: `code/cv_models/models/gan.py`
- Create: `code/cv_models/models/graph.py`

**Key rule:** Each README entry must map to a builder that works for **numpy / torch / tf**.

**Simplification rules:**
- Keep models tiny (fast smoke tests) but preserve the “signature idea” (e.g. residual vs dense concat vs SE vs attention).
- Detection/segmentation output raw heads/logits only.

**Verify:** `pytest -q` passes for numpy and torch; TF tests skipped if TF missing.

---

### Task 5: Add CLI smoke runner + docs

**Files:**
- Create: `code/cv_models/tools/smoke_test.py`
- Create: `code/README.md`
- Modify: `README.md` (add “how to run code” section)

**Commands:**
- NumPy: `PYTHONPATH=code python -m cv_models.tools.smoke_test --backend numpy --model all`
- PyTorch: `PYTHONPATH=code python -m cv_models.tools.smoke_test --backend torch --model all`
- TensorFlow (after install): `PYTHONPATH=code python -m cv_models.tools.smoke_test --backend tf --model all`

**TensorFlow install (example):**
- `pip install tensorflow` (CPU/GPU depending on your environment)

