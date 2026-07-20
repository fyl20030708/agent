# Week01 DeepSeek Bridge → research_agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire DeepSeek (OpenAI-compatible) so `week01` can answer Chinese questions about `31P_transformer` via tools, then thin-migrate the same loop into `research_agent` with shared settings and explicit round budget.

**Architecture:** Keep `week01/tool_calling_demo.py` self-contained. Add `research_agent/llm.py` for env settings only in Phase 2. Both use httpx `POST /chat/completions` + tool messages. No frameworks, no parallel tools, no `_run_async`.

**Tech Stack:** Python 3.10+, httpx, python-dotenv, pytest; DeepSeek Chat Completions API.

**Spec:** `docs/superpowers/specs/2026-07-20-week01-deepseek-bridge-design.md`

## Global Constraints

- Env names stay `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` (never invent `DEEPSEEK_*`)
- Default model: `deepseek-chat`
- Recommended base URL: `https://api.deepseek.com/v1` (so existing path `/chat/completions` resolves correctly)
- `REPO_ROOT` default: `/homes/yli/31P_transformer`
- Do not commit `.env` or real API keys
- week01 must not import `research_agent`
- No live-API tests in CI; unit-test only `load_settings`

## File structure

| File | Responsibility |
|------|----------------|
| `.env.example` | Document DeepSeek-oriented defaults |
| `docs/ROADMAP.md` | Mark zero_week done; Week1 acceptance |
| `README.md` | DeepSeek setup + acceptance commands |
| `week01/tool_calling_demo.py` | Minimal real-API teaching script |
| `src/research_agent/llm.py` | `ClientSettings` + `load_settings()` |
| `src/research_agent/agent.py` | Turn loop using `load_settings` + verbose |
| `tests/test_llm_settings.py` | Unit tests for settings (no network) |

Unchanged: `src/research_agent/tools.py`, `src/research_agent/cli.py`, `tests/test_tools.py`

---

### Task 1: Config + learning docs

**Files:**
- Modify: `.env.example`
- Modify: `docs/ROADMAP.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing
- Produces: documented env contract used by Tasks 2–4

- [ ] **Step 1: Replace `.env.example` contents**

```bash
# DeepSeek (OpenAI-compatible). Prefer /v1 so clients can POST /chat/completions.
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# Research codebase root
REPO_ROOT=/homes/yli/31P_transformer

# Optional: set to 1 to print per-round tool names in research_agent
# AGENT_VERBOSE=0
```

- [ ] **Step 2: Update `docs/ROADMAP.md` stage 0 / milestones**

In「阶段 0」add a completed bullet for async:

```markdown
- [x] Python 异步 `async/await` — 见 `src/zero_week/async_agent_roadmap.py`
```

Change the stage 0 practice line to:

```markdown
**练习：** `week01/tool_calling_demo.py` — DeepSeek + 3 个 function；中文问句读 `REPO_ROOT`。
```

Update milestone checklist:

```markdown
- [x] zero_week：async_agent_roadmap（await / gather / Semaphore / 迷你 loop）
- [ ] Week 1：Tool Calling demo 跑通（DeepSeek + 31P_transformer）
- [ ] Week 2：手写 ReAct loop
```

Keep remaining milestones unchanged.

- [ ] **Step 3: Update `README.md` quick start**

After the venv install block, replace the API comment with:

```markdown
# 配置 DeepSeek（OpenAI 兼容变量名）
cp .env.example .env
# 编辑 .env：填入 OPENAI_API_KEY；确认
# OPENAI_BASE_URL=https://api.deepseek.com/v1
# OPENAI_MODEL=deepseek-chat
# REPO_ROOT=/homes/yli/31P_transformer

# Week 1 验收（真 API）
python week01/tool_calling_demo.py "列出 Generation 目录下的 Python 文件，并读其中一个文件的前 30 行说明它是做什么的"

# 科研助手
python -m research_agent.cli "HybridTransformer 大概在哪个文件？先用工具确认再回答"
```

Also add under 目录结构:

```text
├── src/zero_week/           # 阶段 0：async → 迷你 Agent 引擎
├── docs/superpowers/specs/  # 设计文档
```

- [ ] **Step 4: Commit**

```bash
git add .env.example docs/ROADMAP.md README.md
git commit -m "docs: DeepSeek env defaults and Week1 acceptance path"
```

---

### Task 2: Harden week01 UX (self-contained)

**Files:**
- Modify: `week01/tool_calling_demo.py`

**Interfaces:**
- Consumes: env contract from Task 1
- Produces: `chat(user_message: str, max_rounds: int = 6) -> str` (same signature; adds startup log)

- [ ] **Step 1: Update defaults and logging inside `chat()`**

Replace the env/model block at the start of `chat()` with:

```python
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
```

Keep the existing httpx loop body unchanged (`raise_for_status`, tool append, halted string).

- [ ] **Step 2: Align `__main__` default question (optional clarity)**

```python
if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or (
        "列出 Generation 目录下的 Python 文件，并读其中一个文件的前 30 行说明它是做什么的"
    )
    print(chat(question))
```

- [ ] **Step 3: Manual Phase 1 acceptance (requires user's `.env`)**

Ensure `/homes/yli/agent/.env` exists with a real DeepSeek key (never commit it).

Run:

```bash
cd /homes/yli/agent
python week01/tool_calling_demo.py "列出 Generation 目录下的 Python 文件，并读其中一个文件的前 30 行说明它是做什么的"
```

Expected:
- First line shows `model=deepseek-chat` and `base_url=...`
- Final answer mentions real paths under Generation (or clear tool-derived facts)
- If HTTP 404 on `/chat/completions`, try `OPENAI_BASE_URL=https://api.deepseek.com/v1` vs without `/v1` and update `.env.example` comment to the working value, then re-run

If `OPENAI_API_KEY` missing: expect `RuntimeError: Set OPENAI_API_KEY...`

- [ ] **Step 4: Commit**

```bash
git add week01/tool_calling_demo.py .env.example
git commit -m "feat(week01): DeepSeek defaults and startup model log"
```

---

### Task 3: `research_agent.llm` settings (TDD)

**Files:**
- Create: `src/research_agent/llm.py`
- Create: `tests/test_llm_settings.py`

**Interfaces:**
- Consumes: env vars `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- Produces:
  - `@dataclass class ClientSettings: api_key: str; base_url: str; model: str`
  - `def load_settings() -> ClientSettings`

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm_settings.py`:

```python
import os

import pytest

from research_agent.llm import ClientSettings, load_settings


def test_load_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    s = load_settings()
    assert s == ClientSettings(
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )


def test_load_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    s = load_settings()
    assert s.base_url == "https://api.deepseek.com/v1"
    assert s.model == "deepseek-chat"


def test_load_settings_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        load_settings()
```

- [ ] **Step 2: Run tests — expect fail**

```bash
cd /homes/yli/agent
pytest tests/test_llm_settings.py -v
```

Expected: FAIL with `ModuleNotFoundError: research_agent.llm` (or import error).

- [ ] **Step 3: Implement `src/research_agent/llm.py`**

```python
"""LLM client settings from environment (OpenAI-compatible / DeepSeek)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ClientSettings:
    api_key: str
    base_url: str
    model: str


def load_settings() -> ClientSettings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env (DeepSeek key is fine)")
    return ClientSettings(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_llm_settings.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/research_agent/llm.py tests/test_llm_settings.py
git commit -m "feat(research_agent): add load_settings for DeepSeek-compatible API"
```

---

### Task 4: Wire `research_agent.agent` loop + verbose

**Files:**
- Modify: `src/research_agent/agent.py`
- Modify: `src/research_agent/cli.py` (only if needed to pass verbose via env — prefer reading env inside `ask`)

**Interfaces:**
- Consumes: `load_settings() -> ClientSettings` from Task 3
- Produces: `ask(question: str, max_rounds: int = 8, verbose: bool | None = None) -> str`

- [ ] **Step 1: Update imports and remove inline env reads in `agent.py`**

At top of `src/research_agent/agent.py`, keep existing imports except drop unused `os` if no longer needed for API key (still need `os` for `AGENT_VERBOSE`). Add:

```python
from research_agent.llm import load_settings
```

- [ ] **Step 2: Replace `ask()` body with settings + explicit rounds + verbose**

Replace `ask` with:

```python
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
                print(f"── round {rounds_used}/{max_rounds} ──")

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
                if verbose:
                    print("  (final text)")
                return msg.get("content") or ""

            for call in tool_calls:
                fn = call["function"]
                if verbose:
                    print(f"  tool: {fn['name']}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": _dispatch(fn["name"], fn["arguments"]),
                    }
                )

    return "Max tool rounds reached."
```

Keep `TOOL_SCHEMAS`, `SYSTEM`, `_dispatch` unchanged. Ensure `import os` remains.

- [ ] **Step 3: Run unit tests (no API)**

```bash
pytest tests/test_llm_settings.py tests/test_tools.py -v
```

Expected: all pass (tools may skip if repo missing; settings must pass).

- [ ] **Step 4: Manual Phase 2 acceptance**

```bash
cd /homes/yli/agent
AGENT_VERBOSE=1 python -m research_agent.cli "HybridTransformer 大概在哪个文件？先用工具确认再回答"
```

Expected:
- Prints `── round N/8 ──` and at least one `tool: ...`
- Final answer cites a real path under `REPO_ROOT`

- [ ] **Step 5: Commit**

```bash
git add src/research_agent/agent.py
git commit -m "feat(research_agent): use load_settings and explicit round budget"
```

---

### Task 5: Close ROADMAP checklist after acceptance

**Files:**
- Modify: `docs/ROADMAP.md`

**Interfaces:**
- Consumes: successful Task 2 + Task 4 manual runs
- Produces: milestone boxes reflecting reality

- [ ] **Step 1: Check off Week 1 only after Phase 1+2 both succeeded**

```markdown
- [x] Week 1：Tool Calling demo 跑通（DeepSeek + 31P_transformer）
```

Optionally add under stage 1:

```markdown
- [x] research_agent CLI 能用工具定位 HybridTransformer（Phase 2 验收）
```

Do **not** check Week 2 ReAct unless the user separately verifies `week02/react_loop.py`.

- [ ] **Step 2: Commit**

```bash
git add docs/ROADMAP.md
git commit -m "docs: mark Week1 DeepSeek acceptance complete"
```

---

## Spec coverage self-check

| Spec requirement | Task |
|------------------|------|
| `.env.example` DeepSeek defaults | Task 1 |
| ROADMAP zero_week + Week1 path | Task 1, 5 |
| README DeepSeek + commands | Task 1 |
| week01 self-contained + log + comments | Task 2 |
| Phase 1 acceptance command | Task 2 Step 3 |
| `llm.py` ClientSettings + load_settings | Task 3 |
| week01 does not import research_agent | Task 2 (constraint) |
| agent.py uses settings + rounds + verbose | Task 4 |
| tools/cli/tests_tools unchanged | Tasks 3–4 |
| Phase 2 acceptance | Task 4 Step 4 |
| No live API in CI | Task 3 unit tests only |
| No parallel tools / frameworks | Global constraints |

## Placeholder / consistency check

- No TBD left
- Defaults consistently `https://api.deepseek.com/v1` + `deepseek-chat`
- `load_settings` / `ClientSettings` names match across Task 3–4
