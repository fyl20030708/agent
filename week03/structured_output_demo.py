"""Week 3 (curriculum Week 2): structured JSON output with one repair retry."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(os.getenv("REPO_ROOT", "/homes/yli/31P_transformer"))

SYSTEM = """You are a careful assistant.
Return ONLY a single JSON object (no markdown prose outside JSON) with keys:
  path: string — a file path relative to the repo root
  summary: string — one short sentence about that file
  confidence: number in [0, 1]
Repo root hint: {repo}
If unsure of a real path, still return JSON but use a best-guess relative path and lower confidence.
""".format(repo=REPO_ROOT)


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


def _complete(client: httpx.Client, model: str, messages: list[dict]) -> str:
    resp = client.post(
        "/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"].get("content") or ""


def run_structured(question: str, max_retries: int = 1) -> dict:
    """Ask model for JSON report; one repair retry on contract failure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    print(f"[week03] model={model} base_url={base_url}")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    attempts = max_retries + 1
    last_raw = ""
    with httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    ) as client:
        for i in range(attempts):
            last_raw = _complete(client, model, messages)
            messages.append({"role": "assistant", "content": last_raw})
            obj = extract_json_object(last_raw)
            if obj is not None:
                ok, err = validate_report(obj)
                if ok:
                    return obj
                repair = f"Your JSON failed validation: {err}. Reply with corrected JSON only."
            else:
                repair = "Could not parse JSON. Reply with a single JSON object only."
            if i + 1 >= attempts:
                break
            print(f"[week03] contract failed; retry {i + 1}/{max_retries}: {repair}")
            messages.append({"role": "user", "content": repair})
    raise ValueError(f"Structured output failed after retries. Last raw:\n{last_raw}")


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or (
        "Pick one Python file under Training/ and summarize what it does in one sentence."
    )
    print(json.dumps(run_structured(q), ensure_ascii=False, indent=2))
