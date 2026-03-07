"""
Forward-only, educational implementations of classic CV paper models.

This package intentionally lives under the repo-local `code/` folder.
Use it via:

  PYTHONPATH=code python -m cv_models.tools.smoke_test --backend numpy --model all
"""

from .registry import MODEL_SPECS, README_MODEL_NAMES  # noqa: F401

