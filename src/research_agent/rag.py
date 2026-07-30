"""Thin research-agent adapter over the Week 3 minimal RAG helpers."""

from __future__ import annotations

from typing import Any


def _retrieve(query: str, k: int) -> list[dict[str, Any]]:
    from week04 import rag_lib
    from research_agent.tools import REPO_ROOT

    index = rag_lib.build_index(REPO_ROOT)
    return rag_lib.search(query, index, k=k)


def search_docs(query: str, k: int = 5) -> str:
    """Retrieve relevant Markdown chunks and format them as a tool observation."""
    results = _retrieve(query, k)
    if not results:
        return "(no relevant documentation found)"

    blocks = []
    for index, result in enumerate(results, 1):
        heading = f"#{result['heading']}" if result.get("heading") else ""
        blocks.append(
            f"[{index}] score={result['score']:.2f} path={result['path']}{heading}\n"
            f"{result['text']}"
        )
    return "\n\n".join(blocks)
