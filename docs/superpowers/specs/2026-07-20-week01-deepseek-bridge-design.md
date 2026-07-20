# Week01 DeepSeek Bridge → research_agent

**Date:** 2026-07-20  
**Status:** Approved (brainstorming)  
**Path:** A (real API) → C (research_agent)

## Goal

After finishing `src/zero_week/async_agent_roadmap.py` (mock async agent engine), connect a real LLM so a Chinese terminal question can inspect `/homes/yli/31P_transformer` via tools, then migrate the same loop semantics into `research_agent`.

**Acceptance A:** Agent uses `list_python_files` / `read_file` (and research_agent’s extra tools) against `REPO_ROOT` and answers with real file paths—not fabricated guesses.

## Non-goals (this iteration)

- No new `week03` module
- No Semaphore / parallel tool dispatch / `_run_async` bridge
- No agent frameworks (LangGraph, etc.)
- No API-key-required CI tests (manual acceptance only)
- Do not commit real secrets; only update `.env.example`

## Constraints

| Item | Choice |
|------|--------|
| Provider | DeepSeek via OpenAI-compatible Chat Completions |
| Env vars | Keep `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| Default model | `deepseek-chat` |
| Repo | `REPO_ROOT=/homes/yli/31P_transformer` |
| Approach | Plan 1: harden week01 first, then thin-migrate research_agent |

## Architecture

```text
Phase 1                          Phase 2
────────                         ────────
.env → DeepSeek                  research_agent/llm.py (shared settings)
week01/tool_calling_demo.py      agent.py loop + explicit rounds/Budget comments
ROADMAP + README update          cli unchanged; tools unchanged
```

**Shared data flow (both phases):**

```text
user question
  → messages = [system, user]
  → POST {base_url}/chat/completions  (model, messages, tools)
  → if tool_calls: dispatch → append role=tool → loop
  → if content only: return
  → if max_rounds exhausted: return halted message
```

**Map from async learning → this design:**

| `async_agent_roadmap` | This design |
|----------------------|-------------|
| `run_turn` while loop | `chat()` / `ask()` for-loop |
| `IterationBudget` | `max_rounds` + explicit `rounds_used` (Phase 2) |
| mock `call_llm` | DeepSeek `chat/completions` |
| mock tools | Real filesystem tools under `REPO_ROOT` |

## Phase 1 — week01 real API

### Config

Update `.env.example` with DeepSeek-oriented defaults/comments:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
REPO_ROOT=/homes/yli/31P_transformer
```

If DeepSeek requires a `/v1` suffix on `base_url`, document the working value in `.env.example` after one probe during implementation.

### Code

`week01/tool_calling_demo.py`:

- Keep self-contained (inline env read); do **not** import `research_agent`
- Small UX: log model + base_url (no API key) at start
- Comment that `max_rounds` ≈ `IterationBudget`
- Leave tool set as-is: `list_python_files`, `read_file`, `count_lines`

### Docs

- `docs/ROADMAP.md`: mark zero_week async complete; Week 1 acceptance = DeepSeek + Chinese prompt; link `async_agent_roadmap.py`
- `README.md`: DeepSeek `.env` snippet + Phase 1 acceptance command

### Acceptance

```bash
python week01/tool_calling_demo.py "列出 Generation 目录下的 Python 文件，并读其中一个文件的前 30 行说明它是做什么的"
```

Expect tool use and path-grounded answer.

## Phase 2 — research_agent thin migrate

### Modules

```text
research_agent/
  tools.py   # filesystem only; no HTTP
  llm.py     # NEW: ClientSettings from env
  agent.py   # schemas + dispatch + turn loop
  cli.py     # argv → ask() → print
```

### `llm.py`

- `ClientSettings(api_key, base_url, model)`
- `load_settings()` reads the three `OPENAI_*` vars; raise clear error if key missing
- week01 does **not** depend on this module (avoid coupling)

### `agent.py`

- Use `load_settings()` instead of duplicating env reads
- Name loop counters clearly (`rounds_used` / `max_rounds`); comment ≈ `IterationBudget.consume()`
- Optional verbosity: `ask(..., verbose=False)` or `AGENT_VERBOSE=1` prints per-round tool names
- Still no parallel tools / locks / refund

### Unchanged

- `tools.py` tool surface (already enough for acceptance)
- `cli.py` interface
- `tests/test_tools.py` (no live API tests)

### Acceptance

```bash
python -m research_agent.cli "HybridTransformer 大概在哪个文件？先用工具确认再回答"
```

Expect at least one of `grep_repo` / `list_python_files` / `read_file`, and a path in the answer.

## Error handling (minimum)

| Case | Behavior |
|------|----------|
| Missing API key | Fail fast with message to check `.env` |
| HTTP error | `raise_for_status()`; do not swallow |
| Unknown tool | Return error string to the model |
| Max rounds | Return explicit halted string |

## Success checklist

- [ ] DeepSeek `.env` works with week01 Chinese acceptance prompt
- [ ] `research_agent.cli` same class of question uses tools + cites paths
- [ ] ROADMAP/README describe: async foundation → real API → research_agent

## Implementation order

1. `.env.example` + ROADMAP/README
2. week01 small UX/comments; manual Phase 1 acceptance
3. `research_agent/llm.py` + wire `agent.py`; optional verbose
4. Manual Phase 2 acceptance

## Out of scope follow-ups (later weeks)

- Parallel tool `gather` + Semaphore
- Hermes-style `_run_async` sync/async bridge
- RAG / memory / eval harness
- Cursor MCP / SDK (platform track)
