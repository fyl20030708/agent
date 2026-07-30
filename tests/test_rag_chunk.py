import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week04"))

from rag_lib import chunk_markdown, cosine  # noqa: E402


def test_chunk_markdown_splits_on_headings() -> None:
    text = (
        "# Overview\n"
        "Intro to the project.\n"
        "\n"
        "## Setup\n"
        "Install dependencies first.\n"
        "\n"
        "### Config\n"
        "Edit the config file.\n"
    )
    mtime = 1_700_000_000.0
    chunks = chunk_markdown(text, "docs/guide.md", mtime)

    assert len(chunks) == 3
    assert [c["heading"] for c in chunks] == ["Overview", "Setup", "Config"]
    assert all(c["path"] == "docs/guide.md" for c in chunks)
    assert all(c["mtime"] == mtime for c in chunks)
    assert all(c["text"].strip() for c in chunks)
    assert "Intro to the project." in chunks[0]["text"]
    assert "Install dependencies first." in chunks[1]["text"]
    assert "Edit the config file." in chunks[2]["text"]


def test_chunk_markdown_ignores_deeper_than_h3_headings() -> None:
    text = "#### Deep\nStill part of prior section.\n\n# Top\nBody.\n"
    chunks = chunk_markdown(text, "a.md", 0.0)
    headings = [c["heading"] for c in chunks]
    assert headings == ["", "Top"]
    assert "Still part of prior section." in chunks[0]["text"]
    assert "Body." in chunks[1]["text"]


def test_chunk_markdown_preserves_nonempty_preamble() -> None:
    text = "Project notes before any section.\n\n# First Section\nSection body.\n"
    chunks = chunk_markdown(text, "readme.md", 42.0)

    assert len(chunks) == 2
    assert chunks[0]["heading"] == ""
    assert chunks[0]["path"] == "readme.md"
    assert chunks[0]["mtime"] == 42.0
    assert "Project notes before any section." in chunks[0]["text"]
    assert chunks[1]["heading"] == "First Section"
    assert "Section body." in chunks[1]["text"]


def test_chunk_markdown_windowed_when_no_headings() -> None:
    paragraph = "word " * 200  # 1000 chars of content
    text = paragraph.strip()
    chunks = chunk_markdown(text, "long.md", 99.0)

    assert len(chunks) >= 2
    assert all(c["path"] == "long.md" for c in chunks)
    assert all(c["mtime"] == 99.0 for c in chunks)
    assert all(c["heading"] == "" for c in chunks)
    assert all(c["text"].strip() for c in chunks)
    assert all(len(c["text"]) <= 800 for c in chunks)


def test_cosine_identical_vectors() -> None:
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_orthogonal_vectors() -> None:
    score = cosine([1.0, 0.0], [0.0, 1.0])
    assert math.isclose(score, 0.0, abs_tol=1e-9)


def test_cosine_zero_vector_returns_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine([1.0, 2.0], [0.0, 0.0]) == 0.0
    assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        cosine([1.0, 2.0], [1.0, 2.0, 3.0])
