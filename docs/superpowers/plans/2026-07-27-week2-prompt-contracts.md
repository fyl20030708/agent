# Week 2 Prompt Contracts + Structured Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach and implement “model as unreliable component”: force structured JSON in a generic `week03` demo (with one retry), then migrate `research_agent` so final answers must cite real repo-relative paths or fail/retry.

**Architecture:** Keep `week03/` as a **self-contained** teaching script (OpenAI-compatible chat, optional tools, JSON contract + validator + single retry). Keep domain migration in `research_agent` via a small `contracts.py` helper (path citation check) used by `ask()` after the tool loop. No RAG, no frameworks.

**Tech Stack:** Python 3.10+, httpx, python-dotenv, pytest; same `OPENAI_*` / `REPO_ROOT` env as week01.

**Spec:** `docs/superpowers/specs/2026-07-20-agent-learning-path-design.md` (Week 2 block)

## Global Constraints

- Env names stay `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- Defaults: `https://api.deepseek.com/v1`, `deepseek-chat` (learner may override via `.env`, e.g. ECNU)
- `REPO_ROOT` default: `/homes/yli/31P_transformer`
- `week03` must not import `research_agent` (demo stays self-contained)
- Do not commit `.env` or secrets
- No LangGraph / RAG / MCP / multi-agent in this plan
- Retry budget: **at most one** repair attempt after a failed contract check (unless a step says otherwise)
- Unit-test pure validators without live API; live runs are manual acceptance

## File structure

| File | Responsibility |
|------|----------------|
| `docs/ROADMAP.md` | Point “本周” at Week 2 |
| `week03/structured_output_demo.py` | Generic JSON contract demo + one retry |
| `week03/CONTRACT.md` | system/user/tool schema / retry teaching note |
| `tests/test_structured_parse.py` | Unit tests for JSON extract + schema validate |
| `src/research_agent/contracts.py` | `extract_cited_paths`, `answer_has_repo_path` |
| `tests/test_answer_contract.py` | Unit tests for path citation helper |
| `src/research_agent/agent.py` | After tool loop, enforce path contract + one retry |
| `docs/notes/week2-contract-acceptance.md` | Acceptance checklist template |

---

### Task 1: Point ROADMAP at Week 2

**Files:**
- Modify: `docs/ROADMAP.md` (banner under title)

**Interfaces:**
- Consumes: nothing
- Produces: human-visible Week 2 pointer

- [ ] **Step 1: Update the banner’s “本周” lines**

Replace the current Week 1 “本周” bullet in the top blockquote with:

```markdown
> **本周（Week 2）：** Prompt 契约 + 结构化输出 — `week03/structured_output_demo.py` → 迁入 `research_agent` 路径引用校验。
>
> **已完成（Week 1）：** ReAct / conversation loop — `week02/` + `week01/tool_calling_demo.py`。
```

Keep the existing “近 4 周执行顺序” sentence that links `agent-learning-path-design`.

- [ ] **Step 2: Commit**

```bash
git add docs/ROADMAP.md
git commit -m "docs: point ROADMAP at Week2 prompt-contract focus"
```

---

### Task 2: TDD — JSON extract + schema validate helpers

**Files:**
- Create: `week03/structured_output_demo.py` (helpers only first; full `main` in Task 3)
- Create: `tests/test_structured_parse.py`

**Interfaces:**
- Produces:
  - `extract_json_object(text: str) -> dict | None`
  - `validate_report(obj: dict) -> tuple[bool, str]`  
    Valid iff keys `path` (str, non-empty), `summary` (str, non-empty), `confidence` (int|float in `[0, 1]`)

- [ ] **Step 1: Write failing tests**

Create `tests/test_structured_parse.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week03"))

from structured_output_demo import extract_json_object, validate_report  # noqa: E402


def test_extract_plain_json() -> None:
    text = '{"path": "Training/train.py", "summary": "trainer", "confidence": 0.9}'
    assert extract_json_object(text) == {
        "path": "Training/train.py",
        "summary": "trainer",
        "confidence": 0.9,
    }


def test_extract_json_from_fenced_block() -> None:
    fence = chr(96) * 3  # backtick * 3
    text = (
        "Here you go:\n"
        + fence
        + "json\n"
        + '{"path": "a.py", "summary": "x", "confidence": 0.5}\n'
        + fence
        + "\n"
    )
    obj = extract_json_object(text)
    assert obj is not None
    assert obj["path"] == "a.py"


def test_extract_invalid_returns_none() -> None:
    assert extract_json_object("no json here") is None


def test_validate_report_ok() -> None:
    ok, msg = validate_report(
        {"path": "Training/train.py", "summary": "entry", "confidence": 0.8}
    )
    assert ok is True
    assert msg == ""


def test_validate_report_missing_key() -> None:
    ok, msg = validate_report({"path": "a.py", "summary": "x"})
    assert ok is False
    assert "confidence" in msg.lower() or "missing" in msg.lower()


def test_validate_report_bad_confidence() -> None:
    ok, msg = validate_report(
        {"path": "a.py", "summary": "x", "confidence": 2.0}
    )
    assert ok is False
```

- [ ] **Step 2: Run tests — expect fail**

```bash
cd /homes/yli/agent
pytest tests/test_structured_parse.py -v
```

Expected: import error or failing assertions.

- [ ] **Step 3: Implement helpers in `week03/structured_output_demo.py`**

```python
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
```

- [ ] **Step 4: Re-run tests — expect pass**

```bash
pytest tests/test_structured_parse.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add week03/structured_output_demo.py tests/test_structured_parse.py
git commit -m "feat(week03): add JSON extract/validate helpers with tests"
```

---

### Task 3: Wire `week03` chat demo — contract + one retry

**Files:**
- Modify: `week03/structured_output_demo.py`
- Create: `week03/CONTRACT.md`

**Interfaces:**
- Produces: `run_structured(question: str, max_retries: int = 1) -> dict`
  - Calls chat completions **without tools** (contract focus)
  - System prompt demands **only** JSON object with `path`, `summary`, `confidence`
  - On validate fail: append a user repair message once, then re-call; if still bad, raise `ValueError` with last raw text

- [ ] **Step 1: Add LLM loop + CLI**

Append to `week03/structured_output_demo.py` (reuse week01 env pattern):

```python
import os
from pathlib import Path

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
```

- [ ] **Step 2: Write `week03/CONTRACT.md`**

```markdown
# Prompt 契约（Week 2）

## system / user / tool 分工

| 角色 | 职责 |
|------|------|
| system | 身份、硬规则、输出合同（JSON 字段、禁止废话） |
| user | 具体任务；校验失败时的 **repair** 指令 |
| tool | （本 demo 未用）参数 schema = 工具侧契约；与 JSON 最终答案契约是同一思想 |

## 不可靠组件模型

1. 调用模型  
2. **校验**（parse + schema）  
3. 失败 → **至多一次** repair user 消息  
4. 仍失败 → 对调用方抛错 / 返回 halted（不要假装成功）

## 和 research_agent 迁移的关系

week03 练的是「最终答案 JSON 合同」。  
research_agent 迁的是「最终答案必须引用真实相对路径」——校验换了，**重试回路相同**。
```

- [ ] **Step 3: Manual smoke**

```bash
cd /homes/yli/agent
python week03/structured_output_demo.py
```

Expected: prints `[week03] model=...` and a JSON object with `path` / `summary` / `confidence`. If first reply is prose, expect one `[week03] contract failed; retry ...` then success or `ValueError`.

- [ ] **Step 4: Commit**

```bash
git add week03/structured_output_demo.py week03/CONTRACT.md
git commit -m "feat(week03): structured JSON demo with one contract retry"
```

---

### Task 4: TDD — research_agent path citation contract

**Files:**
- Create: `src/research_agent/contracts.py`
- Create: `tests/test_answer_contract.py`

**Interfaces:**
- Produces:
  - `extract_cited_paths(answer: str) -> list[str]` — find path-like tokens (contain `/` or end with `.py`)
  - `answer_has_repo_path(answer: str, repo_root: Path | None = None) -> bool` — true iff some extracted path exists as a file under `REPO_ROOT` (or `repo_root`)

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from research_agent.contracts import answer_has_repo_path, extract_cited_paths
from research_agent.tools import REPO_ROOT


def test_extract_cited_paths_basic() -> None:
    text = "See Training/train.py and also Generation/foo.py for details."
    paths = extract_cited_paths(text)
    assert "Training/train.py" in paths
    assert "Generation/foo.py" in paths


def test_answer_has_repo_path_true_when_file_exists() -> None:
    # train.py should exist in the 31P repo; skip-friendly assert
    if not (REPO_ROOT / "Training" / "train.py").is_file():
        # fallback: any .py under REPO_ROOT
        sample = next(REPO_ROOT.rglob("*.py"), None)
        assert sample is not None, "REPO_ROOT has no python files"
        rel = str(sample.relative_to(REPO_ROOT))
        assert answer_has_repo_path(f"File is {rel}", REPO_ROOT) is True
    else:
        assert answer_has_repo_path("Look at Training/train.py", REPO_ROOT) is True


def test_answer_has_repo_path_false_for_hallucinated() -> None:
    assert answer_has_repo_path("See Not/A/Real/Path/zzz.py", REPO_ROOT) is False
```

- [ ] **Step 2: Run — expect fail**

```bash
pytest tests/test_answer_contract.py -v
```

- [ ] **Step 3: Implement `src/research_agent/contracts.py`**

```python
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
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_answer_contract.py tests/test_structured_parse.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/research_agent/contracts.py tests/test_answer_contract.py
git commit -m "feat(research_agent): add path-citation answer contract helpers"
```

---

### Task 5: Migrate — enforce path contract in `ask()` with one retry

**Files:**
- Modify: `src/research_agent/agent.py`

**Interfaces:**
- Consumes: `answer_has_repo_path`
- Produces: `ask(...)` still returns `str`; after a final text answer, if contract fails, append a repair user message and continue the **same** round loop (counts toward `max_rounds`); if rounds exhaust still without valid citation, return the last answer prefixed with `[contract-warning] `

- [ ] **Step 1: Strengthen SYSTEM one line**

Append to `SYSTEM`:

```text
Final answers MUST cite at least one existing repo-relative file path (e.g. Training/train.py) that you verified with tools.
```

- [ ] **Step 2: After detecting final text (no tool_calls), run contract**

Replace the “no tool_calls → return content” branch with logic equivalent to:

```python
from research_agent.contracts import answer_has_repo_path

# inside loop, when not tool_calls:
content = msg.get("content") or ""
if answer_has_repo_path(content):
    if verbose:
        print("  (final text, contract ok)")
    return content

# contract fail: spend another round on repair if budget remains
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
```

After the `for` loop ends, return:

```python
return "[contract-warning] Max tool rounds reached without a path-citing answer."
```

(If you still have a last assistant text handy, you may return  
`"[contract-warning]\n" + last_content` instead — pick one and document in the commit message.)

- [ ] **Step 3: Unit tests still pass; manual smoke**

```bash
pytest tests/ -q
AGENT_VERBOSE=1 python -m research_agent.cli "这个仓库的训练入口大概在哪个文件？先用工具确认再回答。"
```

Expected: tools run; final answer contains a real path like `Training/train.py`.  
Optional negative probe (may cost rounds): ask something vague and temporarily weaken tools—skip if too flaky; rely on unit tests for hallucinated paths.

- [ ] **Step 4: Commit**

```bash
git add src/research_agent/agent.py
git commit -m "feat(research_agent): require cited repo path in final answers"
```

---

### Task 6: Acceptance note template

**Files:**
- Create: `docs/notes/week2-contract-acceptance.md`

**Interfaces:**
- Learner fills after manual runs; leave checkboxes unchecked in git

- [ ] **Step 1: Create the note**

Include:
- Title `# Week 2 Acceptance — Prompt contracts`
- Date placeholder
- Commands:
  - `pytest tests/test_structured_parse.py tests/test_answer_contract.py -v`
  - `python week03/structured_output_demo.py`
  - `AGENT_VERBOSE=1 python -m research_agent.cli "这个仓库的训练入口大概在哪个文件？先用工具确认再回答。"`
- Checklist:
  - structured parse tests pass
  - week03 returns valid JSON (possibly after one retry)
  - research_agent answer cites a real relative path
  - I can explain system vs user vs repair vs tool schema using `week03/CONTRACT.md`
- “My 3-sentence summary” placeholder

- [ ] **Step 2: Commit**

```bash
git add docs/notes/week2-contract-acceptance.md
git commit -m "docs: add Week2 contract acceptance checklist template"
```

---

## Spec coverage self-check

| Learning-path Week 2 requirement | Task |
|----------------------------------|------|
| Learn system/user / JSON / retry | Task 3 `CONTRACT.md` + demo |
| Demo `week03/` `{path,summary,confidence}` + one retry | Tasks 2–3 |
| Migrate research_agent path cite or fail/retry | Tasks 4–5 |
| Accept vague Q still prefers tools + real paths | Task 5 smoke + Task 6 |
| ROADMAP Week 2 pointer | Task 1 |
| No RAG / frameworks | Global constraints |

## Placeholder / consistency check

- Retry = one repair only in week03; research_agent repair uses remaining `max_rounds`
- `week03` does not import `research_agent`
- Validators unit-tested without API keys
