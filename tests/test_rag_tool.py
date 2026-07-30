"""Offline tests for the research-agent RAG tool."""

from __future__ import annotations

import pytest

from research_agent import rag


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
