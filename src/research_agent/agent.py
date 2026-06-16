"""Simple tool-calling agent for the 31P research codebase."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from research_agent.tools import REPO_ROOT, TOOL_REGISTRY

load_dotenv()

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_python_files",
            "description": "List .py files under the research repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {"type": "string", "description": "Optional subdirectory, e.g. Training"},
                    "limit": {"type": "integer", "default": 50},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read lines from a file in the research repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string"},
                    "max_lines": {"type": "integer", "default": 100},
                },
                "required": ["relative_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_experiment_dirs",
            "description": "List 2026_* experiment directories at repo root.",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_repo",
            "description": "Search Python files for a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "subdir": {"type": "string", "default": ""},
                    "max_matches": {"type": "integer", "default": 30},
                },
                "required": ["pattern"],
            },
        },
    },
]

SYSTEM = f"""You are a research assistant for 31P MRS deep learning (Hybrid CNN-Transformer).
Repo: {REPO_ROOT}
Always use tools to verify facts from the codebase. Cite file paths in answers."""


def _dispatch(name: str, raw_args: str | dict) -> str:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    return str(fn(**args))


def ask(question: str, max_rounds: int = 8) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    with httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120) as client:
        for _ in range(max_rounds):
            resp = client.post(
                "/chat/completions",
                json={"model": model, "messages": messages, "tools": TOOL_SCHEMAS},
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            messages.append(msg)

            if not msg.get("tool_calls"):
                return msg.get("content") or ""

            for call in msg["tool_calls"]:
                fn = call["function"]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": _dispatch(fn["name"], fn["arguments"]),
                    }
                )

    return "Max tool rounds reached."
