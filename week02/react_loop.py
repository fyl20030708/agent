"""Week 2: hand-rolled ReAct loop (Thought → Action → Observation)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(os.getenv("REPO_ROOT", "/homes/yli/31P_transformer"))

ACTION_PATTERN = re.compile(
    r"Action:\s*(\w+)\((.*?)\)\s*\nObservation:",
    re.DOTALL,
)


def read_file(path: str, max_lines: int = 60) -> str:
    full = REPO_ROOT / path
    if not full.is_file():
        return f"File not found: {path}"
    lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[:max_lines])


def list_dir(path: str = "") -> str:
    full = REPO_ROOT / path if path else REPO_ROOT
    if not full.is_dir():
        return f"Directory not found: {path or '.'}"
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in full.iterdir())
    return "\n".join(entries[:40])


TOOLS = {
    "read_file": read_file,
    "list_dir": list_dir,
}

REACT_PROMPT = f"""You solve tasks about a research codebase using ReAct format.

Available actions (exact syntax):
  Action: read_file("relative/path.py")
  Action: list_dir("Training")
  Action: finish("your final answer to the user")

After each Action line, stop and wait for Observation.
Repo root: {REPO_ROOT}

Example:
Thought: I need to see what's in Training/.
Action: list_dir("Training")
Observation: ...
Thought: ...
Action: finish("...")
"""


def parse_action(text: str) -> tuple[str, dict] | None:
    match = ACTION_PATTERN.search(text + "\nObservation:")
    if not match:
        # Also accept Action: finish("...")
        finish = re.search(r'Action:\s*finish\("(.*)"\)\s*$', text.strip(), re.DOTALL)
        if finish:
            return "finish", {"answer": finish.group(1)}
        return None
    name, raw_args = match.group(1), match.group(2).strip()
    if name == "finish":
        return "finish", {"answer": raw_args.strip('"')}
    # Single string arg tools
    arg = raw_args.strip('"')
    if name == "read_file":
        return name, {"path": arg}
    if name == "list_dir":
        return name, {"path": arg}
    return None


def run_react(question: str, max_steps: int = 8) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    transcript = f"{REACT_PROMPT}\n\nQuestion: {question}\n"

    with httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120) as client:
        for step in range(max_steps):
            resp = client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": transcript + "Thought:"}],
                    "temperature": 0,
                },
            )
            resp.raise_for_status()
            chunk = "Thought:" + resp.json()["choices"][0]["message"]["content"]
            transcript += chunk

            parsed = parse_action(chunk)
            if parsed is None:
                transcript += "\nObservation: Invalid format. Use Action: tool_name(\"arg\")\n"
                continue

            name, args = parsed
            if name == "finish":
                return args["answer"]

            fn = TOOLS.get(name)
            if fn is None:
                obs = f"Unknown action: {name}"
            else:
                obs = fn(**args)
            transcript += f"\nObservation: {obs}\n"
            print(f"[step {step + 1}] {name}({args})")

    return "Max steps reached."


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What training scripts exist under Training/?"
    print(run_react(q))
