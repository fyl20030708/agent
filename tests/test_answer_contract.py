from pathlib import Path

from research_agent.contracts import answer_has_repo_path, extract_cited_paths
from research_agent.tools import REPO_ROOT


def test_extract_cited_paths_basic() -> None:
    text = "See Training/train.py and also Generation/foo.py for details."
    paths = extract_cited_paths(text)
    assert "Training/train.py" in paths
    assert "Generation/foo.py" in paths


def test_answer_has_repo_path_true_when_file_exists() -> None:
    # train.py should exist in the 31P repo; skip-friendly assert
    if not (REPO_ROOT / "Training" / "train.py").is_file():
        # fallback: any .py under REPO_ROOT
        sample = next(REPO_ROOT.rglob("*.py"), None)
        assert sample is not None, "REPO_ROOT has no python files"
        rel = str(sample.relative_to(REPO_ROOT))
        assert answer_has_repo_path(f"File is {rel}", REPO_ROOT) is True
    else:
        assert answer_has_repo_path("Look at Training/train.py", REPO_ROOT) is True


def test_answer_has_repo_path_false_for_hallucinated() -> None:
    assert answer_has_repo_path("See Not/A/Real/Path/zzz.py", REPO_ROOT) is False
