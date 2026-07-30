"""Pure helpers for Week 3 minimal RAG (chunk, cosine, cache, search)."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 100
# Bumped from 1: cache payload gained top-level `embed_model` / `vector_dim`
# keys (see save_cache/load_cache). Old caches lack those keys, which already
# forces a safe rebuild on its own (mismatched embed_model), but the version
# bump makes the schema change explicit.
CHUNK_VERSION = 2
DEFAULT_EMBED_MODEL = "ecnu-embedding-small"
# Named cap on how many inputs go into a single /embeddings HTTP request.
EMBED_BATCH_SIZE = 64
CACHE_PATH = Path(__file__).with_name(".rag_cache.json")
SKIPPED_DIRECTORIES = {".git", ".pytest_cache", "node_modules"}


def _current_embed_model() -> str:
    return os.getenv("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def chunk_markdown(text: str, source_path: str, mtime: float) -> list[dict[str, Any]]:
    sections = _split_by_headings(text)
    if sections:
        chunks: list[dict[str, Any]] = []
        for heading, body in sections:
            body = body.strip()
            if not body:
                continue
            if len(body) <= MAX_CHUNK_CHARS:
                chunks.append(
                    {
                        "path": source_path,
                        "heading": heading,
                        "text": body,
                        "mtime": mtime,
                    }
                )
            else:
                chunks.extend(
                    _window_chunks(body, source_path, mtime, heading=heading)
                )
        return chunks
    return _window_chunks(text, source_path, mtime)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end]))
    return sections


def _window_chunks(
    text: str, source_path: str, mtime: float, heading: str = ""
) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + MAX_CHUNK_CHARS, length)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "path": source_path,
                    "heading": heading,
                    "text": chunk_text,
                    "mtime": mtime,
                }
            )
        if end >= length:
            break
        start = max(start + 1, end - OVERLAP_CHARS)
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vectors must have equal length: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_texts(
    texts: list[str], batch_size: int = EMBED_BATCH_SIZE
) -> list[list[float]]:
    if not texts:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = _current_embed_model()

    vectors: list[list[float]] = []
    with httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    ) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.post(
                "/embeddings",
                json={"model": model, "input": batch},
            )
            response.raise_for_status()
            data = response.json()["data"]
            ordered = sorted(data, key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in ordered)
    return vectors


def load_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {"chunk_version": CHUNK_VERSION, "records": []}
    with cache_path.open(encoding="utf-8") as cache_file:
        return json.load(cache_file)


def save_cache(
    path: str | Path,
    records: list[dict[str, Any]],
    *,
    embed_model: str | None = None,
    vector_dim: int | None = None,
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_version": CHUNK_VERSION,
        "embed_model": embed_model,
        "vector_dim": vector_dim,
        "records": records,
    }
    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, ensure_ascii=False, indent=2)


def needs_reembed(cached_mtime: float, current_mtime: float) -> bool:
    return cached_mtime != current_mtime


def _pair_chunks_with_vectors(
    chunks: list[dict[str, Any]], vectors: list[list[float]]
) -> list[dict[str, Any]]:
    if len(vectors) != len(chunks):
        raise ValueError(
            f"vector count mismatch: expected {len(chunks)}, got {len(vectors)}"
        )
    return [{**chunk, "vector": vector} for chunk, vector in zip(chunks, vectors)]


def build_index(
    repo_root: str | Path,
    embed_fn: Callable[[list[str]], list[list[float]]] = embed_texts,
    embed_model: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    model = embed_model or _current_embed_model()
    cache = load_cache(CACHE_PATH)
    cache_existed = CACHE_PATH.exists()
    cache_matches_schema = (
        cache.get("chunk_version") == CHUNK_VERSION
        and cache.get("embed_model") == model
    )
    cached_records = cache.get("records", []) if cache_matches_schema else []
    cached_by_path: dict[str, list[dict[str, Any]]] = {}
    for record in cached_records:
        cached_by_path.setdefault(record["path"], []).append(record)

    discovered_paths: set[str] = set()
    index: list[dict[str, Any]] = []
    pending_files: list[tuple[str, list[dict[str, Any]]]] = []
    for markdown_path in _discover_markdown(root):
        relative_path = markdown_path.relative_to(root).as_posix()
        discovered_paths.add(relative_path)
        mtime = markdown_path.stat().st_mtime
        cached = cached_by_path.get(relative_path, [])
        if cached and not needs_reembed(cached[0]["mtime"], mtime):
            index.extend(cached)
            continue

        chunks = chunk_markdown(
            markdown_path.read_text(encoding="utf-8"),
            relative_path,
            mtime,
        )
        pending_files.append((relative_path, chunks))

    files_removed = bool(set(cached_by_path) - discovered_paths)
    embedded_any_file = False

    try:
        for relative_path, chunks in pending_files:
            if not chunks:
                continue
            vectors = embed_fn([chunk["text"] for chunk in chunks])
            embedded_any_file = True
            index.extend(_pair_chunks_with_vectors(chunks, vectors))
    except Exception:
        # Only persist on failure if this run made real, keepable progress
        # (a file finished embedding, or stale cached files were dropped).
        # Otherwise leave any existing cache file completely untouched —
        # e.g. a legacy/schema-mismatched cache must not be replaced by an
        # empty v2 payload just because the very first embed call failed.
        if embedded_any_file or files_removed:
            vector_dim = len(index[0]["vector"]) if index else None
            save_cache(CACHE_PATH, index, embed_model=model, vector_dim=vector_dim)
        raise

    # Persist whatever succeeded so far so a retry does not have to re-embed
    # files that already completed this run.
    needs_save = embedded_any_file or files_removed
    if needs_save:
        if index:
            vector_dim = len(index[0]["vector"])
        elif cache_matches_schema:
            vector_dim = cache.get("vector_dim")
        else:
            vector_dim = None
        save_cache(CACHE_PATH, index, embed_model=model, vector_dim=vector_dim)

    return index


def search(
    query: str,
    index: list[dict[str, Any]],
    k: int = 5,
    embed_fn: Callable[[list[str]], list[list[float]]] = embed_texts,
) -> list[dict[str, Any]]:
    if not index or k <= 0:
        return []
    query_vector = embed_fn([query])[0]
    scored = [
        {**record, "score": cosine(query_vector, record["vector"])}
        for record in index
    ]
    return sorted(scored, key=lambda record: record["score"], reverse=True)[:k]


def _discover_markdown(repo_root: Path) -> list[Path]:
    markdown_files: list[Path] = []
    for directory, subdirectories, filenames in os.walk(repo_root):
        subdirectories[:] = sorted(
            name for name in subdirectories if name not in SKIPPED_DIRECTORIES
        )
        base = Path(directory)
        markdown_files.extend(
            base / filename for filename in filenames if filename.endswith(".md")
        )
    return sorted(markdown_files)
