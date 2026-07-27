"""Week 3 (curriculum Week 2): structured JSON output with one repair retry."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort: fenced ```json block, else first {...} span, else whole text."""
    if not text or not text.strip():
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else text.strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate_report(obj: dict[str, Any]) -> tuple[bool, str]:
    required = ("path", "summary", "confidence")
    missing = [k for k in required if k not in obj]
    if missing:
        return False, f"missing keys: {', '.join(missing)}"
    if not isinstance(obj["path"], str) or not obj["path"].strip():
        return False, "path must be a non-empty string"
    if not isinstance(obj["summary"], str) or not obj["summary"].strip():
        return False, "summary must be a non-empty string"
    conf = obj["confidence"]
    if not isinstance(conf, (int, float)) or isinstance(conf, bool):
        return False, "confidence must be a number"
    if not (0.0 <= float(conf) <= 1.0):
        return False, "confidence must be in [0, 1]"
    return True, ""
