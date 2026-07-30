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
CHUNK_VERSION = 1
CACHE_PATH = Path(__file__).with_name(".rag_cache.json")
SKIPPED_DIRECTORIES = {".git", ".pytest_cache", "node_modules"}


def chunk_markdown(text: str, source_path: str, mtime: float) -> list[dict[str, Any]]:
    sections = _split_by_headings(text)
    if sections:
        chunks: list[dict[str, Any]] = []
        for heading, body in sections:
            body = body.strip()
            if not body:
                continue
            chunks.append(
                {
                    "path": source_path,
                    "heading": heading,
                    "text": body,
                    "mtime": mtime,
                }
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


def _window_chunks(text: str, source_path: str, mtime: float) -> list[dict[str, Any]]:
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
                    "heading": "",
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


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("OPENAI_EMBED_MODEL", "ecnu-embedding-small")

    with httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    ) as client:
        response = client.post(
            "/embeddings",
            json={"model": model, "input": texts},
        )
        response.raise_for_status()

    data = response.json()["data"]
    ordered = sorted(data, key=lambda item: item["index"])
    return [item["embedding"] for item in ordered]


def load_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {"chunk_version": CHUNK_VERSION, "records": []}
    with cache_path.open(encoding="utf-8") as cache_file:
        return json.load(cache_file)


def save_cache(path: str | Path, records: list[dict[str, Any]]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunk_version": CHUNK_VERSION, "records": records}
    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, ensure_ascii=False, indent=2)


def needs_reembed(cached_mtime: float, current_mtime: float) -> bool:
    return cached_mtime != current_mtime


def build_index(
    repo_root: str | Path,
    embed_fn: Callable[[list[str]], list[list[float]]] = embed_texts,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    cache = load_cache(CACHE_PATH)
    cached_records = (
        cache.get("records", [])
        if cache.get("chunk_version") == CHUNK_VERSION
        else []
    )
    cached_by_path: dict[str, list[dict[str, Any]]] = {}
    for record in cached_records:
        cached_by_path.setdefault(record["path"], []).append(record)

    file_chunks: list[tuple[bool, list[dict[str, Any]]]] = []
    chunks_to_embed: list[dict[str, Any]] = []
    for markdown_path in _discover_markdown(root):
        relative_path = markdown_path.relative_to(root).as_posix()
        mtime = markdown_path.stat().st_mtime
        cached = cached_by_path.get(relative_path, [])
        if cached and not needs_reembed(cached[0]["mtime"], mtime):
            file_chunks.append((True, cached))
            continue

        chunks = chunk_markdown(
            markdown_path.read_text(encoding="utf-8"),
            relative_path,
            mtime,
        )
        file_chunks.append((False, chunks))
        chunks_to_embed.extend(chunks)

    embedded_vectors = (
        embed_fn([chunk["text"] for chunk in chunks_to_embed])
        if chunks_to_embed
        else []
    )
    vectors = iter(embedded_vectors)
    index: list[dict[str, Any]] = []
    for reused, chunks in file_chunks:
        if reused:
            index.extend(chunks)
            continue
        for chunk in chunks:
            index.append({**chunk, "vector": next(vectors)})

    save_cache(CACHE_PATH, index)
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
