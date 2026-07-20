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
