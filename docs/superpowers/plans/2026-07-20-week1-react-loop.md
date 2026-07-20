# Week 1 ReAct Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Week 1 of the learning path concrete: harden text-ReAct `week02`, document ReAct vs native tool-calling, lightly align `research_agent` verbose loop, and leave a clear acceptance checklist.

**Architecture:** Keep `week02/react_loop.py` as a **self-contained** text-Action ReAct demo (Thought → Action → Observation). Keep `week01` / `research_agent` on native `tool_calls`. Teach the difference in a short markdown note; do not merge the two protocols into one mega-file.

**Tech Stack:** Python 3.10+, httpx, python-dotenv, pytest; OpenAI-compatible Chat Completions (DeepSeek / ECNU / etc. via `OPENAI_*` env).

**Spec:** `docs/superpowers/specs/2026-07-20-agent-learning-path-design.md` (Week 1 block)

## Global Constraints

- Env names stay `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` (never invent `DEEPSEEK_*`)
- Defaults: `OPENAI_BASE_URL=https://api.deepseek.com/v1`, `OPENAI_MODEL=deepseek-chat` (learner may override to ECNU etc. in `.env`)
- `REPO_ROOT` default: `/homes/yli/31P_transformer`
- `week02` must remain self-contained (no import of `research_agent`)
- Do not commit `.env` or API keys
- No LangGraph / RAG / MCP in this plan
- Unit-test only pure helpers (`parse_action`); live ReAct runs are manual acceptance

## File structure

| File | Responsibility |
|------|----------------|
| `docs/ROADMAP.md` | Link learning-path spec; mark Week-1 focus |
| `week02/react_loop.py` | Text-ReAct loop; DeepSeek-aligned defaults; clearer step logs |
| `week02/COMPARISON.md` | ReAct vs week01 `tool_calls` (learner + interview cheat sheet) |
| `tests/test_react_parse.py` | Unit tests for `parse_action` |
| `src/research_agent/agent.py` | Tiny verbose wording aligned to “round ≈ budget consume” |
| `docs/notes/week1-react-acceptance.md` | Acceptance checklist + space for learner notes |

---

### Task 1: Point ROADMAP at the learning-path spec

**Files:**
- Modify: `docs/ROADMAP.md` (top of file)

**Interfaces:**
- Consumes: nothing
- Produces: visible “近 4 周以学习路径设计为准” pointer for humans

- [ ] **Step 1: Insert a short banner after the title**

At the top of `docs/ROADMAP.md`, immediately under `# Agent 开发 12 周学习计划`, insert:

```markdown

> **近 4 周执行顺序：** 以 [agent-learning-path-design](superpowers/specs/2026-07-20-agent-learning-path-design.md) 为准（能力地图：ReAct → 契约 → RAG → Eval）。本文件仍是 12 周总索引。
>
> **本周（Week 1）：** ReAct / conversation loop — `week02/react_loop.py` + 对照 `week01/tool_calling_demo.py`。

```

- [ ] **Step 2: Commit**

```bash
git add docs/ROADMAP.md
git commit -m "docs: point ROADMAP at Week1 ReAct learning-path spec"
```

---

### Task 2: Unit-test and harden `parse_action` (TDD)

**Files:**
- Create: `tests/test_react_parse.py`
- Modify: `week02/react_loop.py` (`parse_action` only if tests force a fix)

**Interfaces:**
- Consumes: existing `parse_action(text: str) -> tuple[str, dict] | None`
- Produces: same signature; reliable parsing for `list_dir`, `read_file`, `finish`

- [ ] **Step 1: Write failing tests**

Create `tests/test_react_parse.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week02"))

from react_loop import parse_action  # noqa: E402


def test_parse_list_dir() -> None:
    text = 'Thought: look\nAction: list_dir("Training")\n'
    assert parse_action(text) == ("list_dir", {"path": "Training"})


def test_parse_read_file() -> None:
    text = 'Thought: read\nAction: read_file("Training/train.py")\n'
    assert parse_action(text) == ("read_file", {"path": "Training/train.py"})


def test_parse_finish() -> None:
    text = 'Thought: done\nAction: finish("Training has train.py")\n'
    assert parse_action(text) == ("finish", {"answer": "Training has train.py"})


def test_parse_invalid_returns_none() -> None:
    assert parse_action("Thought: I am confused\n") is None
```

- [ ] **Step 2: Run tests — expect fail or import issues first**

```bash
cd /homes/yli/agent
pytest tests/test_react_parse.py -v
```

Expected initially: failures or import errors until `sys.path` / parsing works. If all already pass on current code, note GREEN-without-RED in the report and still proceed (do not invent broken parsers).

- [ ] **Step 3: Fix `parse_action` only if a test fails**

Common fix: `finish` with nested quotes / missing trailing newline. Prefer minimal regex adjustments; keep the public return shape `(name, dict)`.

- [ ] **Step 4: Re-run tests — expect pass**

```bash
pytest tests/test_react_parse.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_react_parse.py week02/react_loop.py
git commit -m "test(week02): cover ReAct parse_action helpers"
```

---

### Task 3: Align week02 runtime with week01 env UX

**Files:**
- Modify: `week02/react_loop.py`

**Interfaces:**
- Consumes: `OPENAI_*` env
- Produces: `run_react(question: str, max_steps: int = 8) -> str` (same signature); startup log line

- [ ] **Step 1: Update defaults and startup log in `run_react`**

Replace the env block at the start of `run_react` with:

```python
def run_react(question: str, max_steps: int = 8) -> str:
    """Text-ReAct loop. max_steps ≈ IterationBudget (one LLM call per step)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env (DeepSeek / ECNU key is fine)")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    print(f"[week02-react] model={model} base_url={base_url} repo={REPO_ROOT}")
```

Keep the existing httpx transcript loop body. Ensure step prints stay:

```python
print(f"[step {step + 1}] {name}({args})")
```

- [ ] **Step 2: Chinese default question in `__main__`**

```python
if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Training 目录下有哪些训练相关脚本？用工具确认后再回答。"
    print(run_react(q))
```

- [ ] **Step 3: Manual smoke (requires `.env`)**

```bash
cd /homes/yli/agent
python week02/react_loop.py "Training 目录下有哪些训练相关脚本？用工具确认后再回答。"
```

Expected:
- Startup line with model/base_url
- One or more `[step N] list_dir/read_file(...)`
- Final answer mentions real names under `Training/`

If format failures dominate, tighten `REACT_PROMPT` example only (no protocol change).

- [ ] **Step 4: Commit**

```bash
git add week02/react_loop.py
git commit -m "feat(week02): DeepSeek-aligned defaults and ReAct startup log"
```

---

### Task 4: Write `week02/COMPARISON.md` (ReAct vs tool_calls)

**Files:**
- Create: `week02/COMPARISON.md`

**Interfaces:**
- Consumes: conceptual comparison only
- Produces: interview-ready notes the learner can recite

- [ ] **Step 1: Create the file with this content**

```markdown
# ReAct（week02）vs Tool Calling（week01 / research_agent）

## 一张表

| | week02 文本 ReAct | week01 / research_agent |
|--|-------------------|-------------------------|
| 模型输出 | 自然语言 `Thought` + `Action: name("arg")` | JSON `tool_calls` |
| 解析 | 本地正则 `parse_action` | Provider 已结构化 |
| Observation | 拼回 transcript 字符串 | `role: tool` 消息 |
| 优点 | 教学清晰；不依赖 tools API | 稳定、可并行、工业默认 |
| 缺点 | 格式易碎；难并行 | 黑盒一点，少“思考”痕迹 |

## 和 async_agent_roadmap 的对应

| roadmap | week02 | week01 |
|---------|--------|--------|
| `run_turn` while | `run_react` for-loop | `chat` / `ask` for-loop |
| `call_llm` | completions 无 tools | completions + `tools=` |
| `execute_tool` | `TOOLS[name](**args)` | `dispatch_tool` / `_dispatch` |
| `IterationBudget` | `max_steps` | `max_rounds` |

## 面试一句话

> ReAct 是「推理与行动交错」的控制策略；现代 API 的 tool calling 是同一策略的结构化实现。week02 用文本协议练心智，week01 用工业协议练工程。
```

- [ ] **Step 2: Commit**

```bash
git add week02/COMPARISON.md
git commit -m "docs(week02): compare text ReAct vs native tool_calls"
```

---

### Task 5: Migrate — clarify research_agent verbose as budget consume

**Files:**
- Modify: `src/research_agent/agent.py` (verbose prints only)

**Interfaces:**
- Consumes: existing `ask(..., verbose=...)`
- Produces: clearer logs; no new tools unless smoke proves a gap

- [ ] **Step 1: Adjust verbose labels (minimal)**

In `ask()`, change the verbose round line to:

```python
if verbose:
    print(f"── round {rounds_used}/{max_rounds} (budget consume) ──")
```

And when tools run:

```python
if verbose:
    print(f"  action/tool: {fn['name']}")
```

Do **not** add new tools in this task unless Step 2 fails because a needed tool is missing (prefer using existing `list_python_files` / `read_file` / `grep_repo`).

- [ ] **Step 2: Manual migration smoke**

```bash
cd /homes/yli/agent
AGENT_VERBOSE=1 python -m research_agent.cli "Training 目录下有哪些训练相关脚本？先用工具确认再回答。"
```

Expected: `budget consume` lines + `action/tool:` names; answer cites real paths.

- [ ] **Step 3: Run unit tests**

```bash
pytest tests/test_llm_settings.py tests/test_react_parse.py tests/test_tools.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/research_agent/agent.py
git commit -m "chore(research_agent): label verbose rounds as budget consume"
```

---

### Task 6: Acceptance checklist note

**Files:**
- Create: `docs/notes/week1-react-acceptance.md`

**Interfaces:**
- Consumes: learner fills after manual runs
- Produces: written proof Week 1 is done

- [ ] **Step 1: Create acceptance note**

Create `docs/notes/week1-react-acceptance.md` with:

- Title: `# Week 1 Acceptance — ReAct`
- Field: `**Date:** YYYY-MM-DD`
- Section **Commands run** listing these three commands as a fenced bash block:
  - `pytest tests/test_react_parse.py -v`
  - `python week02/react_loop.py "Training 目录下有哪些训练相关脚本？用工具确认后再回答。"`
  - `AGENT_VERBOSE=1 python -m research_agent.cli "Training 目录下有哪些训练相关脚本？先用工具确认再回答。"`
- Section **Checklist** with unchecked items:
  - `parse_action` unit tests pass
  - week02 printed `[step N] ...` and a grounded answer
  - research_agent verbose showed `budget consume` + tools
  - I can explain ReAct vs tool_calls using `week02/COMPARISON.md`
- Section **My 3-sentence summary** with placeholder text `(write after you finish)`

- [ ] **Step 2: Commit template**

```bash
git add docs/notes/week1-react-acceptance.md
git commit -m "docs: add Week1 ReAct acceptance checklist template"
```

Learner fills the checklist after their own runs (do not fake checkboxes in git).

---

## Spec coverage self-check

| Learning-path Week 1 requirement | Task |
|----------------------------------|------|
| Read/contrast ReAct vs loops | Task 4 COMPARISON + learner paper skim (outside code) |
| Run/refine week02 | Tasks 2–3 |
| Migrate research_agent verbose / budget | Task 5 |
| Accept: explain + multi-round tools | Tasks 4–6 |
| ROADMAP points at learning path | Task 1 |
| No framework/RAG/MCP | Global constraints |

## Placeholder / consistency check

- Defaults match week01: `https://api.deepseek.com/v1` + `deepseek-chat`
- `parse_action` return type unchanged
- week02 stays free of `research_agent` imports
