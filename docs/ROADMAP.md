# Agent 开发 12 周学习计划

> **近 4 周执行顺序：** 以 [agent-learning-path-design](superpowers/specs/2026-07-20-agent-learning-path-design.md) 为准（能力地图：ReAct → 契约 → RAG → Eval）。本文件仍是 12 周总索引。
>
> **本周（Week 1）：** ReAct / conversation loop — `week02/react_loop.py` + 对照 `week01/tool_calling_demo.py`。

## 方向选择

| 方向 | 场景 | 核心技术 |
|------|------|----------|
| A. 通用 Agent 工程 | 聊天机器人、科研助手 | LLM API、Tool Use、RAG、Agent 框架 |
| B. Cursor 平台 Agent | IDE 自动化、CI Agent | MCP、Skills、Hooks、Cursor SDK |
| C. 研究型 Agent | 读论文、跑实验、分析结果 | A + 领域工具 + Eval |

建议：**先 A（4–6 周）→ 再 B 或 C**。

---

## 阶段 0：前置基础（第 1–2 周）

- [x] Python 异步 `async/await` — 见 `src/zero_week/async_agent_roadmap.py`
- HTTP API（httpx）
- Prompt Engineering：system prompt、JSON 结构化输出
- 阅读 [ReAct 论文](https://arxiv.org/abs/2210.03629)

**练习：** `week01/tool_calling_demo.py` — DeepSeek + 3 个 function；中文问句读 `REPO_ROOT`。

---

## 阶段 1：核心概念（第 3–4 周）

| 概念 | 实践 |
|------|------|
| Tool / Function Calling | 封装 3–5 个 Python 函数 |
| ReAct 循环 | `week02/react_loop.py` 手写 loop |
| Memory | SQLite + embedding 存偏好 |
| RAG | 索引 31P_transformer 的 README / doc |
| Planning | 分析 log → 画图 → 报告 |

**框架（选一个）：** LangGraph / Pydantic AI / OpenAI Agents SDK

**毕业项目 v1：** `src/research_agent/` — 读 repo、解释训练脚本。

---

## 阶段 2：工程化（第 5–7 周）

1. Observability（Langfuse / 自建日志）
2. Error handling（重试、超时、fallback）
3. Evaluation（20 条 golden test）
4. Security（工具权限边界）
5. Multi-agent（可选：Planner + Coder + Reviewer）

**毕业项目 v2：** 加日志、eval、Gradio UI。

---

## 阶段 3：Cursor 平台（第 8–9 周）

| 能力 | 项目 |
|------|------|
| MCP | 暴露实验目录：列 run、读 log、对比 metric |
| Cursor SDK | GitHub Action：PR 自动 review 训练脚本 |
| Skills / Hooks | 固化代码规范 |

---

## 阶段 4：进阶（第 10–12 周，按需）

- Agentic RAG
- Browser / Computer Use Agent
- Fine-tuning for tool use
- Multi-modal（31P 光谱图分析）

---

## 里程碑检查表

> **Week 1 验收状态（2026-07-20）：** Tasks 1–4 代码已合并；Phase 1（`week01/tool_calling_demo.py`）与 Phase 2（`research_agent` CLI）的 live DeepSeek 验收待 `.env` 配置 `OPENAI_API_KEY`。

- [x] zero_week：async_agent_roadmap（await / gather / Semaphore / 迷你 loop）
- [ ] Week 1：Tool Calling demo 跑通（DeepSeek + 31P_transformer）
- [ ] research_agent CLI 能用工具定位 HybridTransformer（Phase 2 验收）
- [ ] Week 2：手写 ReAct loop
- [ ] Week 4：research_agent v1 能读 31P_transformer 代码
- [ ] Week 6：RAG 索引 + 10 条 eval
- [ ] Week 8：MCP server 暴露实验工具
- [ ] Week 12：Multi-agent 或领域 Agent 完整 demo
