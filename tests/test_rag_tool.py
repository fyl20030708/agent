"""Offline tests for the research-agent RAG tool."""

from __future__ import annotations

import subprocess
import sys

import pytest

from research_agent import rag
from research_agent.agent import _dispatch


@pytest.fixture(autouse=True)
def _reset_rag_index_cache():
    """Prevent the in-process memoized index (`rag._INDEX_CACHE`) from leaking
    a fake/sentinel value set by one test into a later test that expects a
    real build."""
    rag.clear_index_cache()
    yield
    rag.clear_index_cache()


def test_search_docs_formats_injected_retrieval_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_retrieve(query: str, k: int) -> list[dict[str, object]]:
        calls.append((query, k))
        return [
            {
                "score": 0.8765,
                "path": "Training/MAGNITUDE_TRAINING_GUIDE.md",
                "heading": "Strategy",
                "text": "Train magnitude first, then fine-tune the full model.",
            },
            {
                "score": 0.5,
                "path": "README.md",
                "heading": "",
                "text": "Project overview.",
            },
        ]

    monkeypatch.setattr(rag, "_retrieve", fake_retrieve)

    output = rag.search_docs("magnitude training", k=2)

    assert calls == [("magnitude training", 2)]
    assert output == (
        "[1] score=0.88 path=Training/MAGNITUDE_TRAINING_GUIDE.md#Strategy\n"
        "Train magnitude first, then fine-tune the full model.\n\n"
        "[2] score=0.50 path=README.md\n"
        "Project overview."
    )


def test_dispatch_returns_tool_argument_errors_as_observations() -> None:
    output = _dispatch(
        "read_file",
        {"relative_path": "README.md", "start_line": 10},
    )

    assert output == (
        "error: TypeError: read_file() got an unexpected keyword argument 'start_line'"
    )


def test_search_docs_truncates_oversized_snippets_with_clear_ellipsis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_text = "x" * (rag.MAX_SNIPPET_CHARS + 200)

    def fake_retrieve(query: str, k: int) -> list[dict[str, object]]:
        return [
            {
                "score": 0.9,
                "path": "docs/big.md",
                "heading": "Huge",
                "text": long_text,
            }
        ]

    monkeypatch.setattr(rag, "_retrieve", fake_retrieve)

    output = rag.search_docs("anything", k=1)

    assert len(output) < len(long_text)
    assert "[truncated]" in output
    assert "x" * rag.MAX_SNIPPET_CHARS in output
    assert "x" * (rag.MAX_SNIPPET_CHARS + 1) not in output


def test_retrieve_memoizes_index_in_process_and_reuses_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_index = ["sentinel-index"]
    build_calls: list[object] = []
    search_calls: list[tuple[str, list[object], int]] = []

    class _FakeRagLib:
        @staticmethod
        def build_index(repo_root, **kwargs):
            build_calls.append(repo_root)
            return fake_index

        @staticmethod
        def search(query, index, k=5, **kwargs):
            search_calls.append((query, index, k))
            return []

    monkeypatch.setattr(rag, "_load_rag_lib", lambda: _FakeRagLib)
    rag.clear_index_cache()

    rag._retrieve("first query", 3)
    rag._retrieve("second query", 3)

    assert len(build_calls) == 1, "index must be memoized across repeated _retrieve calls"
    assert [call[1] for call in search_calls] == [fake_index, fake_index]

    rag.clear_index_cache()
    rag._retrieve("third query", 3)

    assert len(build_calls) == 2, "clear_index_cache must force a fresh build on next call"


def test_rag_module_importable_and_loads_week04_lib_from_any_cwd(tmp_path) -> None:
    """Regression test for the src-tree agent only importing week04 from repo root.

    Runs in a fresh subprocess with cwd set far away from the repo so the fix
    cannot rely on '' (cwd) being on sys.path, only on the installed package
    plus a cwd-independent path to week04/rag_lib.py.
    """
    probe = (
        "from research_agent.rag import _load_rag_lib\n"
        "mod = _load_rag_lib()\n"
        "assert hasattr(mod, 'build_index')\n"
        "assert hasattr(mod, 'search')\n"
        "assert hasattr(mod, 'chunk_markdown')\n"
        "print('RAG_IMPORT_OK')\n"
    )
    other_cwd = tmp_path / "far-from-repo"
    other_cwd.mkdir()

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(other_cwd),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "RAG_IMPORT_OK" in result.stdout
