# Agent 通用工程宽底座学习路径（4 周近端 + 12 周总览）

**Date:** 2026-07-20  
**Status:** Approved (brainstorming)  
**Audience:** 准备从事 Agent 开发、先打宽底座、约 3 个月后再选方向的学习者

## Decisions (from brainstorming)

| Topic | Choice |
|-------|--------|
| Career posture | Wide foundation first; pick A (apps) / B (platform) / C (research) after ~12 weeks |
| Time | ~20 hours/week (≈3–4 h/day) |
| Practice vehicle | **C — mixed:** generic demo first, then migrate into `research_agent` |
| Curriculum shape | **Approach 2 — capability map**, not rigid old weekly checklist alone |
| Near-term start | Week-1 block: ReAct / conversation loop (not frameworks, not MCP) |

## Goal

Build portable Agent **application-engineering** fundamentals: tool protocols, agent loops, prompt contracts, minimal RAG, and eval. Use `31P_transformer` + `research_agent` as the integration target without locking a final career track yet.

### 4-week near goal

Complete the next four capability blocks (ReAct → contracts → RAG → eval), each with: learn → generic demo → migrate to `research_agent` → acceptance note.

### Non-goals (this design)

- Locking job track A/B/C now
- LangGraph / multi-agent / MCP / Cursor SDK in the first 4 weeks
- Large repo rewrites unrelated to learning

## Principles

1. **Capability first** — protocols and engineering, not only calling a framework  
2. **Demo → migrate** — generic `week0N` demos, then port into `research_agent`  
3. **Acceptable weekly** — one runnable command + one sentence “what I proved”  
4. **Defer platform track** — MCP / Cursor SDK after week ≥8  

## Current position (do not restart from zero)

| Done | Evidence |
|------|----------|
| Async / mini loop | `src/zero_week/async_agent_roadmap.py` |
| Real API + tool calling | `week01/tool_calling_demo.py` (Generation Q&A verified) |
| Thin research agent | `src/research_agent/` + DeepSeek/OpenAI-compatible env |

**Start here:** ReAct mental model + `week02/react_loop.py`, contrasted with week01 tool-calling loop.

## Capability map (12-week spine)

```text
1. Tool Calling protocol          ✅ week01
2. ReAct / conversation loop      ← next (weeks 1 of near plan)
3. Structured output & prompt contracts
4. Minimal RAG
5. Eval (golden set)
6. One framework rewrite (LangGraph or Pydantic AI)   — weeks 5–6
7. Engineering (retries, logs, permissions)           — week ≥7
8. Optional: MCP / Cursor platform                    — week ≥8
```

After block 6–7, write a **direction checkpoint**: lean A, B, or C.

## Near-term: 4-week plan (~20 h/week)

Weekly rhythm: concepts (~4h) → generic demo (~8h) → migrate to research_agent (~6h) → notes/acceptance (~2h).

### Week 1 — ReAct / conversation loop

- **Learn:** [ReAct paper](https://arxiv.org/abs/2210.03629); map Thought→Action→Observation to `async_agent_roadmap.run_turn` and week01 `tool_calls` loop  
- **Demo:** run/refine `week02/react_loop.py`; document text-Action vs native `tool_calls`  
- **Migrate:** research_agent verbose rounds ≈ IterationBudget consume; add tools only if needed  
- **Accept:** explain ReAct vs pure tool-calling; CLI shows multi-round tools on a 31P question  
- **Proves:** agent main loop (interview core)

### Week 2 — Prompt contracts + structured output

- **Learn:** system/user split; JSON / tool parameter schemas; retry on bad shape  
- **Demo:** `week03/` script forcing `{path, summary, confidence}` with one retry  
- **Migrate:** research_agent answers must cite repo-relative paths; otherwise fail/retry  
- **Accept:** vague questions still prefer tools + real paths  
- **Proves:** treating the model as an unreliable component

### Week 3 — Minimal RAG

- **Learn:** chunk → embed → retrieve → context; when RAG vs `read_file`  
- **Demo:** `week04/` index README/docs under `REPO_ROOT`  
- **Migrate:** `search_docs` (or equivalent) with clear split vs `grep_repo` / `read_file`  
- **Accept:** same question via RAG-only vs read_file-only, compare citations  
- **Proves:** knowledge ingestion keyword for jobs

### Week 4 — Minimal eval + direction note

- **Learn:** golden set; pass/fail criteria  
- **Demo:** `evals/` ≥10 fixed questions; script runs research_agent, records pass/fail  
- **Migrate:** classify failures (wrong tool / hallucinated path / round budget)  
- **Accept:** results table + 1-page note “lean A/B/C or keep wide?”  
- **Proves:** measurable improvement habit

### Explicitly deferred past week 4

- Framework rewrite, multi-agent, MCP/SDK, rich memory, fine-tuning

## Repo layout (learning)

```text
agent/
├── docs/ROADMAP.md                 # 12-week index; link this spec at top
├── docs/superpowers/specs/
│   └── 2026-07-20-agent-learning-path-design.md
├── src/zero_week/                  # done
├── week01/                         # done
├── week02/                         # week-1 home (ReAct)
├── week03/                         # week-2 structured output (create)
├── week04/                         # week-3 minimal RAG (create)
├── evals/                          # week-4 golden + runner
└── src/research_agent/             # migration target each block
```

Generic demos stay in `week0N/`; domain integration stays in `research_agent`.

## Success criteria (end of 4 weeks)

- [ ] Can explain ReAct vs tool-calling loop with pointers to files in this repo  
- [ ] Vague questions still tend to cite real relative paths (or fail closed)  
- [ ] At least one doc-retrieval path demos  
- [ ] ≥10 golden evals with pass/fail log  
- [ ] ≥3 migrations into research_agent across the four weeks  
- [ ] One-page direction note (A/B/C or continue wide)

## Relationship to existing `docs/ROADMAP.md`

| Old ROADMAP | This design |
|-------------|-------------|
| Stage 0 leftovers (prompt, paper) | Folded into near weeks 1–2 |
| Stage 1 (ReAct / memory / RAG / framework) | First 4 weeks: ReAct + contracts + RAG + eval; **memory + framework → weeks 5–6** |
| Stage 2 engineering | After week ≥7 |
| Stage 3 Cursor/MCP | After week ≥8 |

`ROADMAP.md` remains the 12-week index. **Next 4 weeks follow this spec** to avoid dual progress tracking.

## Immediate next actions (after user approves this written spec)

1. Link this file from `docs/ROADMAP.md` (short “近 4 周以本设计为准”)  
2. Invoke writing-plans for **Week 1 ReAct block only** (week02 + notes + small research_agent touch)  
3. Learner can start today: ReAct paper skim + read `week02/react_loop.py` vs `week01/tool_calling_demo.py`

## Out of scope follow-ups

- Implementing week03/week04/evals before Week 1 plan is executed  
- Pushing to remote / PR (orthogonal to learning design)
