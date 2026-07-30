# Week 3 Acceptance — Minimal RAG

**Date:** YYYY-MM-DD

## Environment used

| Variable | Value (no secrets) |
|----------|-------------------|
| `OPENAI_API_KEY` | set in `.env` (do not paste) |
| `OPENAI_BASE_URL` | `https://chat.ecnu.edu.cn/open/api/v1` |
| `OPENAI_MODEL` | `ecnu-max` |
| `OPENAI_EMBED_MODEL` | `ecnu-embedding-small` (code default; optional override) |
| `REPO_ROOT` | `/homes/yli/31P_transformer` |

Chat and embed both use the same ECNU-compatible base URL this week (DeepSeek chat-only bases lack `/embeddings`).

## Commands run

Run from repo root unless noted. First `rag_demo` invocation builds or loads the cache.

```bash
# Build index + answer (first run embeds; second run reuses cache)
cd week04
python rag_demo.py "How is magnitude training configured?"

# Search-only (no chat call)
python rag_demo.py "How is magnitude training configured?" --search-only

# Answer with custom top-k
python rag_demo.py "magnitude training strategy" -k 3

# Agent with verbose tool log (from repo root)
cd ..
AGENT_VERBOSE=1 python -m research_agent.cli "Explain the magnitude training strategy and cite the doc."
```

## Same question: RAG-only vs read_file-only

Use one shared question, e.g. *“How is magnitude training configured?”*

### RAG-only (`week04/rag_demo.py`)

```text
(paste top-k hits + Answer section from:)
python rag_demo.py "How is magnitude training configured?"
```

### read_file-only (agent or manual)

```text
(paste output from a run that uses read_file only — no search_docs — e.g. agent with a prompt that forbids search_docs, or direct read_file on a known path)
```

### Citation / quality comparison

| Aspect | RAG-only | read_file-only |
|--------|----------|----------------|
| Paths cited | (fill in) | (fill in) |
| Grounding | retrieval snippets vs full file | exact file contents |
| Notes | (fill in: breadth, precision, misses) | (fill in) |

## When RAG vs direct file tools (one paragraph)

Write after your live session: when would you use `search_docs` (semantic search across Markdown) vs `read_file` (exact file/line contents) vs `grep_repo` (exact symbol/string in `.py`)? Mention at least one case where RAG is the wrong tool.

## Verified evidence (implementation session)

Recorded from live smoke and tests before this note — not placeholders.

- **week04 first run:** `[week04] embedded 618 chunks; cache=week04/.rag_cache.json`
- **week04 second run:** `[week04] reused document cache; cache=week04/.rag_cache.json`
- **Agent live run (3 rounds):** round 1 `search_docs` → round 2 `read_file` → round 3 `final` (contract ok); final answer cited `Training/MAGNITUDE_TRAINING_GUIDE.md`
- **Tests before this docs change:** `pytest -q` → **35 passed**

## Checklist (spec acceptance)

- [ ] `week04` builds or loads cache and answers with at least one real markdown path citation
- [ ] I can explain in one paragraph: RAG vs `read_file` vs `grep_repo` (see section above)
- [ ] Same question compared RAG-only vs read_file-only (quality / citation notes filled in)
- [ ] `search_docs` appears in research_agent tool list and is used on a doc-oriented question (verbose log)
