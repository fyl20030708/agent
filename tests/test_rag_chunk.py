import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week04"))

import rag_lib  # noqa: E402
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


def test_cache_roundtrip_preserves_records_and_chunk_version(tmp_path: Path) -> None:
    cache_path = tmp_path / "index.rag_cache.json"
    records = [
        {
            "path": "docs/guide.md",
            "heading": "Overview",
            "text": "Project overview.",
            "mtime": 123.0,
            "vector": [0.25, 0.75],
        }
    ]

    rag_lib.save_cache(cache_path, records)

    assert rag_lib.load_cache(cache_path) == {
        "chunk_version": rag_lib.CHUNK_VERSION,
        "records": records,
    }


def test_needs_reembed_only_when_mtime_changes() -> None:
    assert rag_lib.needs_reembed(123.0, 123.0) is False
    assert rag_lib.needs_reembed(123.0, 124.0) is True


def test_build_index_reuses_unchanged_vectors_and_embeds_changed_markdown_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    unchanged = repo_root / "unchanged.md"
    changed = repo_root / "changed.md"
    ignored = repo_root / "ignored.py"
    unchanged.write_text("# Keep\nSame text.\n", encoding="utf-8")
    changed.write_text("# Change\nFirst text.\n", encoding="utf-8")
    ignored.write_text("# Not markdown\nDo not index.\n", encoding="utf-8")
    monkeypatch.setattr(rag_lib, "CACHE_PATH", tmp_path / "cache.json")

    embed_calls: list[list[str]] = []

    def stub_embed(texts: list[str]) -> list[list[float]]:
        embed_calls.append(texts)
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

    first_index = rag_lib.build_index(repo_root, embed_fn=stub_embed)
    assert len(embed_calls) == 1
    assert len(embed_calls[0]) == 2
    assert {record["path"] for record in first_index} == {
        "changed.md",
        "unchanged.md",
    }

    unchanged_vector = next(
        record["vector"]
        for record in first_index
        if record["path"] == "unchanged.md"
    )
    second_index = rag_lib.build_index(repo_root, embed_fn=stub_embed)
    assert len(embed_calls) == 1
    assert next(
        record["vector"]
        for record in second_index
        if record["path"] == "unchanged.md"
    ) == unchanged_vector

    changed.write_text("# Change\nUpdated text.\n", encoding="utf-8")
    changed_mtime = changed.stat().st_mtime + 1.0
    os.utime(changed, (changed_mtime, changed_mtime))

    third_index = rag_lib.build_index(repo_root, embed_fn=stub_embed)
    assert len(embed_calls) == 2
    assert embed_calls[-1] == ["Updated text."]
    assert next(
        record["vector"]
        for record in third_index
        if record["path"] == "unchanged.md"
    ) == unchanged_vector


def test_search_returns_top_k_in_descending_cosine_order() -> None:
    index = [
        {"path": "low.md", "heading": "Low", "text": "low", "vector": [0.0, 1.0]},
        {"path": "high.md", "heading": "High", "text": "high", "vector": [1.0, 0.0]},
        {"path": "middle.md", "heading": "Middle", "text": "middle", "vector": [1.0, 1.0]},
    ]

    results = rag_lib.search(
        "query",
        index,
        k=2,
        embed_fn=lambda texts: [[1.0, 0.0]],
    )

    assert [result["path"] for result in results] == ["high.md", "middle.md"]
    assert results[0]["score"] > results[1]["score"]
