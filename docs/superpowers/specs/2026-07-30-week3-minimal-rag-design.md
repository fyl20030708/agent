# Week 3 Minimal RAG Design

**Date:** 2026-07-30  
**Status:** Draft (awaiting user review)  
**Parent:** [agent-learning-path-design](2026-07-20-agent-learning-path-design.md) — Week 3 block  
**Audience:** Same learner track (demo → migrate into `research_agent`)

## Decisions (from brainstorming)

| Topic | Choice |
|-------|--------|
| Embedding backend | **B — OpenAI-compatible `/embeddings`** via existing `OPENAI_*` env |
| Embedding model | `ecnu-embedding-small` (ECNU catalog; env override `OPENAI_EMBED_MODEL`) |
| Index cache | **Disk cache** under `week04/.rag_cache.json` (mtime-aware rebuild) |
| Corpus | Markdown under `REPO_ROOT` (README, `advise/`, `Training/*.md`, `memory/`, etc.) |
| Demo home | `week04/` (curriculum Week 3; numbering matches existing week01–03 layout) |
| Migrate target | `research_agent` tool `search_docs` + prompt guidance vs `read_file` / `grep_repo` |

## Goal

Teach and ship a **minimal RAG loop**: chunk → embed → retrieve → inject context → answer with citations. Contrast when to use RAG vs direct file tools. Prove “knowledge ingestion” as a portable Agent skill.

### Non-goals

- Vector DB (Chroma/FAISS/Pinecone) as a hard dependency
- Hybrid BM25 + dense rerank (ECNU has `ecnu-rerank`; defer)
- Indexing all `.py` source (scope = docs/markdown for this week)
- Agentic RAG / multi-hop retrieval frameworks
- Changing Week 2 path-citation contract semantics

## Architecture

```text
REPO_ROOT/**/*.md
    → discover + chunk (path, heading, text, mtime)
    → embed via POST {OPENAI_BASE_URL}/embeddings
    → cache: week04/.rag_cache.json  (chunks + vectors + source mtimes)
    → query embed → cosine top-k
    → prompt with retrieved snippets + citations
    → LLM answer

research_agent:
    search_docs(query, k) → same retrieve (shared or thin re-export of week04 helpers)
    vs read_file / grep_repo (exact file / regex)
```

### Pipeline stages

1. **Discover** — walk `REPO_ROOT` for `*.md`, skip obvious junk (`.pytest_cache`, `.git`, `node_modules` if any).
2. **Chunk** — split on markdown headings (`#`–`###`) when present; otherwise fixed-size windows (~400–800 chars) with small overlap. Each chunk stores: `path` (repo-relative), `heading` (optional), `text`, `mtime`.
3. **Embed** — batch or sequential calls to `/embeddings` with `model=OPENAI_EMBED_MODEL` (default `ecnu-embedding-small`). Store float vectors alongside chunks.
4. **Cache** — write `week04/.rag_cache.json`. On rebuild: if file mtime unchanged and chunking version matches, reuse vectors; else re-embed that file’s chunks. Full rebuild if schema/`chunk_version` bumps.
5. **Retrieve** — embed query; cosine similarity; return top-k with scores.
6. **Answer (demo)** — system/user prompt includes retrieved blocks as context; ask model to cite `path` (and heading if any).
7. **Migrate** — expose `search_docs` as a tool Observation source (same grounding rule as ReAct: tool returns facts; model does not invent retrieval hits).

### Env

| Variable | Role | Default |
|----------|------|---------|
| `OPENAI_API_KEY` | Auth (existing) | required |
| `OPENAI_BASE_URL` | API root (existing) | ECNU / DeepSeek learner override |
| `OPENAI_MODEL` | Chat model for answer step | `ecnu-max` / learner |
| `OPENAI_EMBED_MODEL` | Embedding model | `ecnu-embedding-small` |
| `REPO_ROOT` | Corpus root | `/homes/yli/31P_transformer` |

**Compatibility note:** DeepSeek chat endpoints do **not** offer `/embeddings`. Learners on DeepSeek-only must either switch embed base to ECNU (or another embed provider) or set a separate embed URL later. This week’s primary verified path is ECNU (`ecnu-embedding-small` probed OK).

### Cache file policy

- Path: `week04/.rag_cache.json` (gitignored; regenerated locally).
- May grow large (vectors × chunks); do not commit.
- Optional tiny fixture cache under `tests/fixtures/` only if needed for offline tests (prefer mock vectors in unit tests instead).

## File structure

| File | Responsibility |
|------|----------------|
| `docs/ROADMAP.md` | Point “本周” at Week 3 / `week04` |
| `week04/rag_demo.py` | Self-contained: build/load index, search, answer CLI |
| `week04/RAG.md` | Teaching note: RAG vs `read_file` / `grep_repo`; cache; embed env |
| `week04/.rag_cache.json` | Local cache (gitignored) |
| `tests/test_rag_chunk.py` | Chunk + cosine unit tests (no live API) |
| `src/research_agent/rag.py` or reuse helpers | Index load + `search_docs` implementation |
| `src/research_agent/tools.py` | Register `search_docs` |
| `src/research_agent/agent.py` | Tool schema + system prompt tool split |
| `docs/notes/week3-rag-acceptance.md` | Acceptance checklist (RAG-only vs read_file-only) |
| `.gitignore` | Ignore `week04/.rag_cache.json` (and similar) |

**Import rule:** `week04` must not import `research_agent`. Migration may import shared math/helpers from a small module **or** duplicate a thin copy initially; prefer extracting pure functions (`chunk_markdown`, `cosine`, cache load/save) into something both can use without circular imports—e.g. keep pure helpers inside `week04` and copy-or-import carefully. Recommended for this repo: put **pure** chunk/cosine/cache I/O in `week04/rag_lib.py` (or top of `rag_demo.py`), and have `research_agent` **duplicate the tool wrapper** calling the same logic via import from `week04` only if packaging allows; otherwise copy the pure functions into `src/research_agent/rag.py` once (teaching cost OK for one week).

**Preferred layout for this learner repo:**

- `week04/rag_lib.py` — discover, chunk, embed client, cache, search (importable)
- `week04/rag_demo.py` — CLI `__main__`
- `src/research_agent/rag.py` — thin adapter: load cache path configurable, `search_docs` string formatter for tool Observation
- Agent registers tool; does not reimplement cosine.

If import path from package to `week04` is awkward, duplicate `search` formatter in research_agent and share only via documented “same algorithm” — but **prefer making `week04` importable** as a module path on `PYTHONPATH`/repo root (same as other week scripts).

## Tool contract (`search_docs`)

```text
search_docs(query: str, k: int = 5) -> str
```

Returns a human-readable Observation, e.g.:

```text
[1] score=0.82 path=Training/MAGNITUDE_TRAINING_GUIDE.md#Overview
<snippet>

[2] ...
```

Prompt guidance (system or tool descriptions):

| Need | Prefer |
|------|--------|
| Conceptual / “how does X work” / design notes | `search_docs` |
| Exact file contents / line-level | `read_file` |
| Exact symbol / string match in `.py` | `grep_repo` |

## Testing strategy

- **Unit (no network):** chunk boundaries; cosine order; empty corpus; cache schema load/save with fake vectors.
- **Optional mock httpx:** embed client returns fixed dims.
- **Manual live:** build index once; run `python week04/rag_demo.py "…"`, then CLI with `search_docs` via research_agent on the same question; fill acceptance note.

## Acceptance (learner)

1. `week04` builds or loads cache and answers with at least one real markdown path citation.
2. Can explain in one paragraph: RAG vs `read_file` vs `grep_repo`.
3. Same question compared RAG-only vs read_file-only (quality / citation notes in acceptance template).
4. `search_docs` appears in research_agent tool list and is used on a doc-oriented question (verbose log).

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Embed provider ≠ chat provider | Document `OPENAI_EMBED_MODEL`; primary path ECNU |
| Cache stale after doc edits | mtime invalidation per source file |
| Huge cache / many md files | Cap max files or max chunks in demo; document knobs |
| Model ignores retrieved context | Prompt: “answer using CONTEXT; cite paths; if missing say so” |
| API key in `.env` | Never commit; rotate if exposed |

## Spec self-review

- [x] No TBD placeholders for core decisions
- [x] Aligns with parent Week 3 block (demo `week04` → migrate `search_docs`)
- [x] Scope excludes vector DB / full `.py` index / rerank
- [x] Cache path and gitignore called out
- [x] DeepSeek embed gap documented
