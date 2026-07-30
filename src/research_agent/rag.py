"""Thin research-agent adapter over the Week 3 minimal RAG helpers.

Loads `week04/rag_lib.py` by absolute file path (resolved from this file's own
location) instead of `from week04 import rag_lib`. The latter only works when
the repo root happens to be on `sys.path` (e.g. cwd == repo root), which broke
whenever the agent — or a caller of this package — ran from any other
directory. Resolving via `__file__` makes the import cwd-independent while
still requiring no packaging changes and no circular import (week04 never
imports research_agent).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_WEEK04_RAG_LIB_PATH = Path(__file__).resolve().parents[2] / "week04" / "rag_lib.py"
_RAG_LIB_MODULE_NAME = "research_agent._week04_rag_lib"

# Cap on how many characters of a retrieved chunk go into a single formatted
# Observation block, independent of upstream chunk sizing (see MAX_CHUNK_CHARS
# in rag_lib) — a defense-in-depth guard against oversized tool output.
MAX_SNIPPET_CHARS = 500
_TRUNCATION_SUFFIX = "\n... [truncated]"

# In-process memoized index so search_docs does not reload/rebuild the (large)
# on-disk cache on every call within the same process.
_INDEX_CACHE: list[dict[str, Any]] | None = None


def _load_rag_lib() -> ModuleType:
    """Import week04/rag_lib.py by absolute path, cached in sys.modules.

    Works from any current working directory and does not require `week04`
    to be an importable package on `sys.path`.
    """
    cached = sys.modules.get(_RAG_LIB_MODULE_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        _RAG_LIB_MODULE_NAME, _WEEK04_RAG_LIB_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load week04 rag_lib from {_WEEK04_RAG_LIB_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RAG_LIB_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def clear_index_cache() -> None:
    """Drop the in-process memoized index.

    Forces the next `_retrieve` call to rebuild/reload via `rag_lib.build_index`.
    Exposed for tests (and for callers that know the on-disk corpus changed).
    """
    global _INDEX_CACHE
    _INDEX_CACHE = None


def _retrieve(query: str, k: int) -> list[dict[str, Any]]:
    global _INDEX_CACHE
    from research_agent.tools import REPO_ROOT

    rag_lib = _load_rag_lib()
    if _INDEX_CACHE is None:
        _INDEX_CACHE = rag_lib.build_index(REPO_ROOT)
    return rag_lib.search(query, _INDEX_CACHE, k=k)


def _format_snippet(text: str) -> str:
    if len(text) <= MAX_SNIPPET_CHARS:
        return text
    return text[:MAX_SNIPPET_CHARS].rstrip() + _TRUNCATION_SUFFIX


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
            f"{_format_snippet(result['text'])}"
        )
    return "\n\n".join(blocks)
