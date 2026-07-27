"""Answer contracts for research_agent (path citation, etc.)."""

from __future__ import annotations

import re
from pathlib import Path

from research_agent.tools import REPO_ROOT

# Paths like Training/train.py or foo/bar/baz.py
_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)+\S+?\.(?:py|md|txt|yaml|yml|json|sh))"
)


def extract_cited_paths(answer: str) -> list[str]:
    found = [m.group("path").rstrip(".,);:]") for m in _PATH_RE.finditer(answer or "")]
    # de-dupe, keep order
    out: list[str] = []
    for p in found:
        if p not in out:
            out.append(p)
    return out


def answer_has_repo_path(answer: str, repo_root: Path | None = None) -> bool:
    root = repo_root or REPO_ROOT
    for rel in extract_cited_paths(answer):
        candidate = root / rel
        if candidate.is_file():
            return True
    return False
