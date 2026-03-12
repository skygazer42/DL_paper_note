
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from cv_models.registry import MODEL_SPECS


def test_root_readme_lists_local_backend_links_for_every_model() -> None:
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")

    missing: list[str] = []
    for model_id in MODEL_SPECS:
        expected_links = (
            f"[numpy](code/numpy_models/{model_id}.py)",
            f"[pytorch](code/pytorch_models/{model_id}.py)",
            f"[tensorflow](code/tensorflow_models/{model_id}.py)",
        )
        missing.extend(link for link in expected_links if link not in readme_text)

    assert not missing, f"README is missing backend links: {missing}"


def test_root_readme_does_not_leave_bare_backend_words_on_model_lines() -> None:
    readme_lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    bare_backend = re.compile(r"(?<!\[)\b(?:numpy|pytorch|tensorflow)\b(?!\]\()", re.IGNORECASE)

    offenders = [line for line in readme_lines if line.lstrip().startswith("- **") and bare_backend.search(line)]
    assert not offenders, f"README still contains bare backend markers: {offenders}"
