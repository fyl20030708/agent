"""Pure helpers for Week 3 minimal RAG (chunk, cosine, cache, search)."""

from __future__ import annotations

import re
from typing import Any

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 100


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
