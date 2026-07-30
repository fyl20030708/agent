"""Tools for inspecting the 31P_transformer research repository."""

from __future__ import annotations

import os
from pathlib import Path

from research_agent.rag import search_docs

REPO_ROOT = Path(os.getenv("REPO_ROOT", "/homes/yli/31P_transformer"))


def list_python_files(subdir: str = "", limit: int = 50) -> str:
    root = REPO_ROOT / subdir if subdir else REPO_ROOT
    if not root.is_dir():
        return f"Directory not found: {root}"
    files = sorted(str(p.relative_to(REPO_ROOT)) for p in root.rglob("*.py"))
    if len(files) > limit:
        return "\n".join(files[:limit]) + f"\n... and {len(files) - limit} more"
    return "\n".join(files) or "(no Python files)"


def read_file(relative_path: str, max_lines: int = 100) -> str:
    path = REPO_ROOT / relative_path
    if not path.is_file():
        return f"File not found: {relative_path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    numbered = "\n".join(f"{i + 1:4d}| {line}" for i, line in enumerate(lines[:max_lines]))
    if len(lines) > max_lines:
        numbered += f"\n... ({len(lines) - max_lines} more lines)"
    return numbered


def list_experiment_dirs(limit: int = 20) -> str:
    """List top-level experiment checkpoint dirs (2026_* pattern)."""
    dirs = sorted(
        p.name for p in REPO_ROOT.iterdir() if p.is_dir() and p.name.startswith("2026_")
    )
    if len(dirs) > limit:
        return "\n".join(dirs[:limit]) + f"\n... and {len(dirs) - limit} more"
    return "\n".join(dirs) or "(no 2026_* experiment dirs at repo root)"


def grep_repo(pattern: str, subdir: str = "", max_matches: int = 30) -> str:
    import re

    root = REPO_ROOT / subdir if subdir else REPO_ROOT
    if not root.is_dir():
        return f"Directory not found: {root}"
    rx = re.compile(pattern)
    hits: list[str] = []
    for py in root.rglob("*.py"):
        try:
            for i, line in enumerate(py.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    rel = py.relative_to(REPO_ROOT)
                    hits.append(f"{rel}:{i}: {line.strip()[:120]}")
                    if len(hits) >= max_matches:
                        return "\n".join(hits) + "\n..."
        except OSError:
            continue
    return "\n".join(hits) or f"No matches for /{pattern}/"


TOOL_REGISTRY = {
    "list_python_files": list_python_files,
    "read_file": read_file,
    "list_experiment_dirs": list_experiment_dirs,
    "grep_repo": grep_repo,
    "search_docs": search_docs,
}
