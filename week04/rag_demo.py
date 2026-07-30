"""Week 3 minimal RAG demo: retrieve Markdown context and answer with citations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from rag_lib import CACHE_PATH, build_index, embed_texts, search

load_dotenv()

REPO_ROOT = Path(os.getenv("REPO_ROOT", "/homes/yli/31P_transformer"))


def build_or_load() -> list[dict[str, Any]]:
    embedded_count = 0

    def tracked_embed(texts: list[str]) -> list[list[float]]:
        nonlocal embedded_count
        embedded_count += len(texts)
        return embed_texts(texts)

    index = build_index(REPO_ROOT, embed_fn=tracked_embed)
    if embedded_count:
        print(f"[week04] embedded {embedded_count} chunks; cache={CACHE_PATH}")
    else:
        print(f"[week04] reused document cache; cache={CACHE_PATH}")
    return index


def _format_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for rank, hit in enumerate(hits, start=1):
        heading = hit["heading"] or "(no heading)"
        blocks.append(
            f"[{rank}] path={hit['path']}#{heading}\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def _chat_answer(query: str, hits: list[dict[str, Any]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    context = _format_context(hits)
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using only the supplied CONTEXT. "
                "Cite supporting sources with their real relative path and heading "
                "exactly as `path#heading`. Do not invent paths or headings. "
                "If the CONTEXT does not cover the answer, say so."
            ),
        },
        {
            "role": "user",
            "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context}",
        },
    ]
    with httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    ) as client:
        response = client.post(
            "/chat/completions",
            json={"model": model, "messages": messages, "temperature": 0},
        )
        response.raise_for_status()
    return response.json()["choices"][0]["message"].get("content") or ""


def answer(query: str, k: int = 5) -> tuple[list[dict[str, Any]], str]:
    index = build_or_load()
    hits = search(query, index, k=k)
    return hits, _chat_answer(query, hits)


def _print_hits(hits: list[dict[str, Any]]) -> None:
    for rank, hit in enumerate(hits, start=1):
        heading = hit["heading"] or "(no heading)"
        print(
            f"[{rank}] {hit['path']}#{heading} "
            f"score={hit['score']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Question to retrieve and answer")
    parser.add_argument("-k", type=int, default=5, help="Number of retrieval hits")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Print ranked hits without calling the chat model",
    )
    args = parser.parse_args()
    query = " ".join(args.question)

    if args.search_only:
        hits = search(query, build_or_load(), k=args.k)
        _print_hits(hits)
        return

    hits, response = answer(query, k=args.k)
    _print_hits(hits)
    print(f"\nAnswer:\n{response}")


if __name__ == "__main__":
    main()
