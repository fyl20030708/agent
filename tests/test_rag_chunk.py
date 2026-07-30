import json
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


def test_chunk_markdown_windows_oversized_heading_sections() -> None:
    long_body = ("word " * 300).strip()  # ~1500 chars, well over MAX_CHUNK_CHARS
    text = f"# Big Section\n{long_body}\n"
    mtime = 5.0
    chunks = chunk_markdown(text, "big.md", mtime)

    assert len(chunks) >= 2
    assert all(c["heading"] == "Big Section" for c in chunks)
    assert all(c["path"] == "big.md" for c in chunks)
    assert all(c["mtime"] == mtime for c in chunks)
    assert all(len(c["text"]) <= rag_lib.MAX_CHUNK_CHARS for c in chunks)
    assert all(c["text"].strip() for c in chunks)
    # reassembled text should still cover the original content (windows overlap)
    assert "word" in chunks[0]["text"] and "word" in chunks[-1]["text"]


def test_chunk_markdown_keeps_small_heading_sections_as_single_chunk() -> None:
    text = "# Small\nJust a short body.\n"
    chunks = chunk_markdown(text, "small.md", 1.0)
    assert len(chunks) == 1
    assert chunks[0]["heading"] == "Small"


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


def test_cache_roundtrip_preserves_records_chunk_version_and_embed_model(
    tmp_path: Path,
) -> None:
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

    rag_lib.save_cache(cache_path, records, embed_model="fake-embed-model", vector_dim=2)

    assert rag_lib.load_cache(cache_path) == {
        "chunk_version": rag_lib.CHUNK_VERSION,
        "embed_model": "fake-embed-model",
        "vector_dim": 2,
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
        embed_calls.append(list(texts))
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

    # Two files need embedding on first build; embedding happens per-file so a
    # later failure on one file cannot discard an already-embedded file (see
    # test_build_index_persists_progress_when_embed_fails_partway).
    first_index = rag_lib.build_index(repo_root, embed_fn=stub_embed, embed_model="fake-model")
    assert len(embed_calls) == 2
    assert {record["path"] for record in first_index} == {
        "changed.md",
        "unchanged.md",
    }

    unchanged_vector = next(
        record["vector"]
        for record in first_index
        if record["path"] == "unchanged.md"
    )
    second_index = rag_lib.build_index(repo_root, embed_fn=stub_embed, embed_model="fake-model")
    assert len(embed_calls) == 2
    assert next(
        record["vector"]
        for record in second_index
        if record["path"] == "unchanged.md"
    ) == unchanged_vector

    changed.write_text("# Change\nUpdated text.\n", encoding="utf-8")
    changed_mtime = changed.stat().st_mtime + 1.0
    os.utime(changed, (changed_mtime, changed_mtime))

    third_index = rag_lib.build_index(repo_root, embed_fn=stub_embed, embed_model="fake-model")
    assert len(embed_calls) == 3
    assert embed_calls[-1] == ["Updated text."]
    assert next(
        record["vector"]
        for record in third_index
        if record["path"] == "unchanged.md"
    ) == unchanged_vector


def test_build_index_skips_save_when_nothing_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "doc.md").write_text("# S\nBody.\n", encoding="utf-8")
    monkeypatch.setattr(rag_lib, "CACHE_PATH", tmp_path / "cache.json")

    save_calls: list[int] = []
    original_save_cache = rag_lib.save_cache

    def spy_save_cache(*args, **kwargs):
        save_calls.append(1)
        return original_save_cache(*args, **kwargs)

    monkeypatch.setattr(rag_lib, "save_cache", spy_save_cache)

    def stub_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    rag_lib.build_index(repo_root, embed_fn=stub_embed, embed_model="fake-model")
    assert len(save_calls) == 1

    rag_lib.build_index(repo_root, embed_fn=stub_embed, embed_model="fake-model")
    assert len(save_calls) == 1, "no files/model/version changed; save must be skipped"


def test_build_index_rebuilds_when_embed_model_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "doc.md").write_text("# S\nBody text.\n", encoding="utf-8")
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(rag_lib, "CACHE_PATH", cache_path)

    embed_calls: list[list[str]] = []

    def stub_embed(texts: list[str]) -> list[list[float]]:
        embed_calls.append(list(texts))
        return [[1.0, 2.0] for _ in texts]

    rag_lib.build_index(repo_root, embed_fn=stub_embed, embed_model="model-a")
    assert len(embed_calls) == 1

    rag_lib.build_index(repo_root, embed_fn=stub_embed, embed_model="model-a")
    assert len(embed_calls) == 1, "same embed model must reuse cached vectors"

    rag_lib.build_index(repo_root, embed_fn=stub_embed, embed_model="model-b")
    assert len(embed_calls) == 2, "embed model change must force a full rebuild"

    cache = rag_lib.load_cache(cache_path)
    assert cache["embed_model"] == "model-b"
    assert cache["vector_dim"] == 2


def test_build_index_rebuilds_when_cache_predates_embed_model_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache written before this fix (no embed_model/vector_dim keys) must be
    treated as stale rather than silently trusted, even without a chunk_version bump."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "doc.md").write_text("# S\nBody.\n", encoding="utf-8")
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(rag_lib, "CACHE_PATH", cache_path)
    cache_path.write_text(
        json.dumps({"chunk_version": rag_lib.CHUNK_VERSION, "records": []}),
        encoding="utf-8",
    )

    embed_calls: list[list[str]] = []

    def stub_embed(texts: list[str]) -> list[list[float]]:
        embed_calls.append(list(texts))
        return [[0.1] for _ in texts]

    rag_lib.build_index(repo_root, embed_fn=stub_embed, embed_model="model-a")
    assert len(embed_calls) == 1


def test_build_index_persists_progress_when_embed_fails_partway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "a.md").write_text("# A\nAlpha body.\n", encoding="utf-8")
    (repo_root / "b.md").write_text("# B\nBeta body.\n", encoding="utf-8")
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(rag_lib, "CACHE_PATH", cache_path)

    calls: list[list[str]] = []

    def flaky_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        if len(calls) == 2:
            raise RuntimeError("network blip")
        return [[1.0, 0.0] for _ in texts]

    with pytest.raises(RuntimeError, match="network blip"):
        rag_lib.build_index(repo_root, embed_fn=flaky_embed, embed_model="model-a")

    cache = rag_lib.load_cache(cache_path)
    assert {record["path"] for record in cache["records"]} == {"a.md"}

    second_index = rag_lib.build_index(repo_root, embed_fn=flaky_embed, embed_model="model-a")
    assert len(calls) == 3
    assert calls[-1] == ["Beta body."]
    assert {record["path"] for record in second_index} == {"a.md", "b.md"}


def test_embed_texts_batches_requests_and_preserves_response_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        def __init__(self, data: list[dict]) -> None:
            self._data = data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": self._data}

    class _FakeHttpxClient:
        def __init__(self, *args, **kwargs) -> None:
            self.calls: list[list[str]] = []

        def __enter__(self) -> "_FakeHttpxClient":
            return self

        def __exit__(self, *exc_info) -> bool:
            return False

        def post(self, path: str, json: dict) -> _FakeResponse:
            batch = json["input"]
            self.calls.append(list(batch))
            # Return entries index-shuffled (reversed) to prove the caller
            # reorders by the response "index" field rather than trusting order.
            data = [
                {"index": position, "embedding": [float(len(text)), float(position)]}
                for position, text in enumerate(batch)
            ]
            data.reverse()
            return _FakeResponse(data)

    fake_clients: list[_FakeHttpxClient] = []

    def fake_client_factory(*args, **kwargs) -> _FakeHttpxClient:
        client = _FakeHttpxClient(*args, **kwargs)
        fake_clients.append(client)
        return client

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(rag_lib.httpx, "Client", fake_client_factory)

    texts = [f"chunk-{i}" for i in range(5)]
    vectors = rag_lib.embed_texts(texts, batch_size=2)

    client = fake_clients[0]
    assert [len(batch) for batch in client.calls] == [2, 2, 1]

    expected: list[list[float]] = []
    for start in range(0, len(texts), 2):
        sub_batch = texts[start : start + 2]
        for local_index, text in enumerate(sub_batch):
            expected.append([float(len(text)), float(local_index)])
    assert vectors == expected


def test_embed_texts_empty_input_requires_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert rag_lib.embed_texts([]) == []


def test_search_returns_empty_list_for_empty_index() -> None:
    assert rag_lib.search("q", [], k=5, embed_fn=lambda texts: [[1.0]]) == []


def test_search_returns_empty_list_when_k_not_positive() -> None:
    index = [{"path": "a.md", "heading": "", "text": "x", "vector": [1.0]}]
    assert rag_lib.search("q", index, k=0, embed_fn=lambda texts: [[1.0]]) == []
    assert rag_lib.search("q", index, k=-1, embed_fn=lambda texts: [[1.0]]) == []


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
