"""Simple tool-calling agent for the 31P research codebase."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from research_agent.contracts import answer_has_repo_path
from research_agent.llm import load_settings
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
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search Markdown documentation for conceptual or design information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM = f"""You are a research assistant for 31P MRS deep learning (Hybrid CNN-Transformer).
Repo: {REPO_ROOT}
Always use tools to verify facts from the codebase. Cite file paths in answers.
Use search_docs for concepts and design questions from documentation.
Use read_file for exact file contents and grep_repo for exact symbols or strings in Python.
Final answers MUST cite at least one existing repo-relative file path (e.g. Training/train.py) that you verified with tools."""


def _dispatch(name: str, raw_args: str | dict) -> str:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    return str(fn(**args))


def ask(question: str, max_rounds: int = 8, verbose: bool | None = None) -> str:
    """Conversation turn loop. Each LLM call consumes one round (≈ IterationBudget)."""
    settings = load_settings()
    if verbose is None:
        verbose = os.getenv("AGENT_VERBOSE", "0") in {"1", "true", "True"}

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    with httpx.Client(
        base_url=settings.base_url,
        headers={"Authorization": f"Bearer {settings.api_key}"},
        timeout=120,
    ) as client:
        for rounds_used in range(1, max_rounds + 1):
            if verbose:
                print(f"── round {rounds_used}/{max_rounds} (budget consume) ──")

            resp = client.post(
                "/chat/completions",
                json={
                    "model": settings.model,
                    "messages": messages,
                    "tools": TOOL_SCHEMAS,
                },
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                content = msg.get("content") or ""
                if answer_has_repo_path(content):
                    if verbose:
                        print("  (final text, contract ok)")
                    return content

                if verbose:
                    print("  (final text failed path contract; requesting repair)")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your answer did not cite any existing file path under the repo. "
                            "Use tools if needed, then answer again and include a real relative path."
                        ),
                    }
                )
                continue

            for call in tool_calls:
                fn = call["function"]
                if verbose:
                    print(f"  action/tool: {fn['name']}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": _dispatch(fn["name"], fn["arguments"]),
                    }
                )

    return "[contract-warning] Max tool rounds reached without a path-citing answer."
