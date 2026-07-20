"""Week 1: minimal tool-calling demo without agent frameworks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(os.getenv("REPO_ROOT", "/homes/yli/31P_transformer"))


def list_python_files(subdir: str = "") -> str:
    """List .py files under REPO_ROOT, optionally scoped to a subdirectory."""
    root = REPO_ROOT / subdir if subdir else REPO_ROOT
    if not root.is_dir():
        return f"Directory not found: {root}"
    files = sorted(str(p.relative_to(REPO_ROOT)) for p in root.rglob("*.py"))
    return "\n".join(files[:50]) or "(no Python files found)"


def read_file(relative_path: str, max_lines: int = 80) -> str:
    """Read the first N lines of a file under REPO_ROOT."""
    path = REPO_ROOT / relative_path
    if not path.is_file():
        return f"File not found: {relative_path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    head = "\n".join(f"{i + 1:4d}| {line}" for i, line in enumerate(lines[:max_lines]))
    suffix = f"\n... ({len(lines) - max_lines} more lines)" if len(lines) > max_lines else ""
    return head + suffix


def count_lines(relative_path: str) -> str:
    """Count lines in a file under REPO_ROOT."""
    path = REPO_ROOT / relative_path
    if not path.is_file():
        return f"File not found: {relative_path}"
    return str(len(path.read_text(encoding="utf-8", errors="replace").splitlines()))


TOOLS = {
    "list_python_files": list_python_files,
    "read_file": read_file,
    "count_lines": count_lines,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_python_files",
            "description": "List Python files in the research repo, optionally under a subdirectory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {
                        "type": "string",
                        "description": "Subdirectory relative to repo root, e.g. 'Generation' or 'Training'.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the first lines of a file in the research repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string", "description": "Path relative to repo root."},
                    "max_lines": {"type": "integer", "description": "Max lines to return.", "default": 80},
                },
                "required": ["relative_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_lines",
            "description": "Count lines in a file in the research repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string", "description": "Path relative to repo root."},
                },
                "required": ["relative_path"],
            },
        },
    },
]

SYSTEM_PROMPT = f"""You are a research code assistant for a 31P MRS transformer project.
The repo root is: {REPO_ROOT}
Use tools to inspect files before answering. Be concise and cite file paths."""


def dispatch_tool(name: str, arguments: str | dict) -> str:
    fn = TOOLS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    return fn(**args)


def chat(user_message: str, max_rounds: int = 6) -> str:
    """Tool-calling loop. max_rounds ≈ IterationBudget from async_agent_roadmap."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env (DeepSeek key is fine)")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    print(f"[week01] model={model} base_url={base_url} repo={REPO_ROOT}")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    with httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120) as client:
        for _ in range(max_rounds):
            resp = client.post(
                "/chat/completions",
                json={"model": model, "messages": messages, "tools": TOOL_SCHEMAS},
            )
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return message.get("content") or ""

            for call in tool_calls:
                fn = call["function"]
                result = dispatch_tool(fn["name"], fn["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )

    return "Max tool rounds reached without a final answer."


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or (
        "列出 Generation 目录下的 Python 文件，并读其中一个文件的前 30 行说明它是做什么的"
    )
    print(chat(question))
