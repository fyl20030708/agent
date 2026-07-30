# Week 3 Minimal RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a minimal RAG pipeline (chunk → embed → cache → retrieve → answer) as a self-contained `week04/` demo, then migrate a `search_docs` tool into `research_agent` with a clear split vs `read_file` / `grep_repo`.

**Architecture:** Pure helpers (discover/chunk/cosine/cache) live in `week04/rag_lib.py`; the CLI wrapper in `week04/rag_demo.py`. `research_agent` gets a thin `rag.py` that loads the same cache and formats a tool Observation, registered as `search_docs`. Embeddings via OpenAI-compatible `/embeddings` (`ecnu-embedding-small`). Cache is a gitignored JSON keyed by source-file mtime.

**Tech Stack:** Python 3.10+, httpx, python-dotenv, pytest. No vector DB, no new heavy deps. Same `OPENAI_*` / `REPO_ROOT` env as prior weeks, plus `OPENAI_EMBED_MODEL`.

**Spec:** `docs/superpowers/specs/2026-07-30-week3-minimal-rag-design.md`

## Global Constraints

- Env names stay `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`; add `OPENAI_EMBED_MODEL` (default `ecnu-embedding-small`).
- Chat default stays learner override (ECNU `ecnu-max` locally; repo default `deepseek-chat` in code is fine).
- `REPO_ROOT` default `/homes/yli/31P_transformer`.
- `week04` must NOT import `research_agent`.
- Corpus = Markdown (`*.md`) under `REPO_ROOT` only this week (no `.py` indexing).
- Retrieval = cosine over dense embeddings; top-k default 5.
- Cache file `week04/.rag_cache.json` is **gitignored**; never commit it or `.env`.
- Unit tests must run offline (no live API): use pure functions + injected/mocked vectors.
- At most **one** embed call path; batch inputs where the API allows a list.
- Do not commit secrets; if a key is exposed, tell the user to rotate.

## File structure

| File | Responsibility |
|------|----------------|
| `.gitignore` | Add `week04/.rag_cache.json` (and `*.rag_cache.json`) |
| `docs/ROADMAP.md` | Point “本周” banner at Week 3 / `week04` RAG |
| `week04/rag_lib.py` | Pure + IO helpers: discover, chunk_markdown, cosine, embed client, cache load/save, search |
| `week04/rag_demo.py` | CLI `__main__`: build/load index, `search`, `answer` |
| `week04/RAG.md` | Teaching note: RAG vs read_file/grep_repo; cache; embed env |
| `tests/test_rag_chunk.py` | Offline unit tests: chunking, cosine, cache round-trip, retrieve ordering (mock vectors) |
| `src/research_agent/rag.py` | Load cache + `search_docs(query, k) -> str` Observation formatter |
| `src/research_agent/tools.py` | Register `search_docs` in `TOOL_REGISTRY` |
| `src/research_agent/agent.py` | Add `search_docs` tool schema + system-prompt tool-split guidance |
| `docs/notes/week3-rag-acceptance.md` | Acceptance checklist (RAG-only vs read_file-only) |

**Cross-module rule:** `research_agent/rag.py` imports pure helpers from `week04.rag_lib` (repo root is already on path for week scripts). If that import proves awkward at execution time, copy the pure `cosine` + cache-load functions into `rag.py` and note the duplication — do NOT create a circular import.

---

### Task 1: Point ROADMAP at Week 3 + gitignore cache

**Files:** `docs/ROADMAP.md`, `.gitignore`

- [ ] Update the top “本周” banner in `docs/ROADMAP.md` to Week 3 (Minimal RAG): `week04/rag_demo.py` → migrate `search_docs`; mark Week 2 done.
- [ ] Add to `.gitignore`:
  ```
  week04/.rag_cache.json
  *.rag_cache.json
  ```
- [ ] Commit: `docs: point ROADMAP at Week3 minimal RAG; ignore rag cache`.

**Acceptance:** ROADMAP shows Week 3 focus; `git status` would not list the cache file once created.

---

### Task 2: Chunking + cosine (TDD, pure functions)

**Files:** `week04/rag_lib.py`, `tests/test_rag_chunk.py`

- [ ] Write failing tests in `tests/test_rag_chunk.py`:
  - `chunk_markdown(text, source_path, mtime)` splits on `#`/`##`/`###` headings; each chunk has `path`, `heading`, `text` (non-empty), `mtime`.
  - Text with no headings → fixed-size windowed chunks (≤ ~800 chars), no empty chunks.
  - `cosine(a, b)` returns 1.0 for identical vectors, ~0 for orthogonal, handles zero-vector without ZeroDivisionError (return 0.0).
- [ ] Run tests; confirm they fail (module/functions missing).
- [ ] Implement `chunk_markdown` and `cosine` in `week04/rag_lib.py` (pure, no network).
- [ ] Run tests; confirm pass.
- [ ] Commit: `feat(week04): markdown chunking + cosine with unit tests`.

**Acceptance:** `pytest tests/test_rag_chunk.py -q` passes; functions are pure and importable.

---

### Task 3: Embed client + mtime-aware cache

**Files:** `week04/rag_lib.py`, `tests/test_rag_chunk.py`

- [ ] Add failing tests:
  - `save_cache(path, records)` then `load_cache(path)` round-trips chunks + vectors + `chunk_version`.
  - `needs_reembed(cached_mtime, current_mtime)` / cache-merge logic reuses vectors when mtime unchanged, drops when changed. (Test the pure decision function with fake data — no API.)
- [ ] Run; confirm fail.
- [ ] Implement in `rag_lib.py`:
  - `embed_texts(texts: list[str]) -> list[list[float]]` — POST `{OPENAI_BASE_URL}/embeddings`, `model=OPENAI_EMBED_MODEL` (default `ecnu-embedding-small`), returns vectors in input order. (Not unit-tested live; keep it a thin, injectable function.)
  - `load_cache` / `save_cache` (JSON) with `chunk_version` constant.
  - `build_index(repo_root, embed_fn=embed_texts)` — discover `*.md`, chunk, reuse cached vectors by `(path, mtime)`, embed only new/changed, write cache. `embed_fn` param makes it testable with a stub.
- [ ] Run; confirm pass (using stub `embed_fn` in tests).
- [ ] Commit: `feat(week04): embed client + mtime-aware JSON cache`.

**Acceptance:** cache round-trip test passes; `build_index` re-embeds only changed files (verified with stub embed_fn call count).

---

### Task 4: Retrieve + demo CLI (live smoke)

**Files:** `week04/rag_lib.py`, `week04/rag_demo.py`, `tests/test_rag_chunk.py`

- [ ] Add failing test: `search(query_vec, index, k)` returns top-k records sorted by descending cosine (use fake vectors so order is deterministic).
- [ ] Implement `search(query, index, k=5, embed_fn=...)` in `rag_lib.py` (embed query, cosine vs all chunk vectors, return top-k with scores).
- [ ] Implement `week04/rag_demo.py`:
  - `build_or_load()` → index via cache.
  - `answer(query, k)` → format retrieved blocks into a CONTEXT prompt; call chat model (reuse env like week03); instruct “answer using CONTEXT, cite `path` (and heading); if not covered, say so”.
  - `__main__`: `python week04/rag_demo.py "question"` prints top-k hits (path#heading + score) then the answer. Support `--search-only` to print hits without the LLM call.
- [ ] Run unit tests; confirm pass.
- [ ] **Live smoke (manual):** `python week04/rag_demo.py "How is magnitude training configured?"` builds cache and returns hits + an answer citing a real `.md` path. Record the command + first-run cache creation in the acceptance note.
- [ ] Commit: `feat(week04): cosine retrieve + RAG answer CLI`.

**Acceptance:** unit tests pass; live run returns ≥1 real markdown path; second run reuses cache (faster / no re-embed logs).

---

### Task 5: Teaching note RAG.md

**Files:** `week04/RAG.md`

- [ ] Write `week04/RAG.md` covering:
  - Pipeline diagram chunk → embed → cache → retrieve → answer.
  - Decision table: `search_docs` (concepts/design) vs `read_file` (exact contents) vs `grep_repo` (exact symbol).
  - Cache: what’s stored, mtime invalidation, why gitignored.
  - Env: `OPENAI_EMBED_MODEL=ecnu-embedding-small`; DeepSeek has no `/embeddings` (compat note).
  - Grounding parallel to ReAct: retrieval hits are Observations (facts from tool), model must not invent them.
- [ ] Commit: `docs(week04): RAG teaching note`.

**Acceptance:** note explains when NOT to use RAG and the embed-provider caveat.

---

### Task 6: Migrate `search_docs` into research_agent

**Files:** `src/research_agent/rag.py`, `src/research_agent/tools.py`, `src/research_agent/agent.py`

- [ ] Create `src/research_agent/rag.py`:
  - Import pure helpers from `week04.rag_lib` (or copy `cosine`/cache-load if import is awkward — document choice in a top comment).
  - `search_docs(query: str, k: int = 5) -> str` → load/build index (cache under `REPO_ROOT`-scoped or `week04/.rag_cache.json`), return a formatted Observation string: `[i] score=.. path=..#heading\n<snippet>`.
- [ ] Register `"search_docs": search_docs` in `TOOL_REGISTRY` (`tools.py`).
- [ ] In `agent.py`:
  - Add a `search_docs` entry to `TOOL_SCHEMAS` (params: `query` required, `k` int default 5).
  - Extend `SYSTEM` with tool-split guidance: use `search_docs` for conceptual/design questions from docs; `read_file` for exact file contents; `grep_repo` for exact symbols. Keep existing path-citation contract.
- [ ] Run full suite: `pytest -q` (existing 21+ still pass; no live calls in tests).
- [ ] **Live smoke (manual):** `AGENT_VERBOSE=1 python -m research_agent.cli "Explain the magnitude training strategy and cite the doc."` shows `action/tool: search_docs` in a round and final answer cites a real path.
- [ ] Commit: `feat(research_agent): add search_docs RAG tool + prompt split`.

**Acceptance:** `search_docs` callable via agent tool loop; verbose log shows it used; final answer still satisfies path contract; unit suite green.

---

### Task 7: Acceptance checklist + wrap

**Files:** `docs/notes/week3-rag-acceptance.md`

- [ ] Create acceptance template with sections:
  - Env used (chat + embed model, base_url).
  - Commands run (build, search-only, answer, agent verbose).
  - Same question: RAG-only vs read_file-only — paste both, note citation/quality diff.
  - One-paragraph “when RAG vs direct file tools”.
  - Checkboxes mirroring spec acceptance (4 items).
- [ ] Commit: `docs: Week3 RAG acceptance checklist template`.
- [ ] Final whole-branch review: run `pytest -q`; skim diff; confirm cache not tracked; confirm no secrets committed.

**Acceptance:** learner can fill the note after one live session; suite green; cache untracked.

---

## Testing summary

| Layer | How |
|-------|-----|
| chunk_markdown / cosine / search order / cache round-trip | offline pytest with fake vectors |
| embed client | thin `embed_fn`, injected; not asserted live |
| demo answer + agent search_docs | manual live smoke, recorded in acceptance note |

## Commit cadence

One commit per task (7 total). Keep `week04` self-contained; migration commit is separate so a reviewer can gate demo vs integration independently.
