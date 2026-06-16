# Agent 开发学习项目

从零学习 LLM Agent 工程，最终目标是做一个能读懂 `31P_transformer` 科研代码与实验数据的助手。

## 目录结构

```
agent/
├── docs/ROADMAP.md          # 12 周学习计划
├── week01/                  # 阶段 0–1：Tool Calling 手写 demo
├── week02/                  # ReAct 循环
├── src/research_agent/      # 毕业项目：科研代码助手
└── tests/
```

## 快速开始

```bash
cd /homes/yli/agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 配置 API Key（OpenAI 或兼容接口）
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# Week 1：Tool Calling demo
python week01/tool_calling_demo.py "列出 Generation 目录下的 Python 文件"

# 科研助手（需配置 REPO_ROOT）
python -m research_agent.cli "解释 HybridTransformer 的整体结构"
```

## 关联项目

- 科研代码库：`/homes/yli/31P_transformer`（通过 `REPO_ROOT` 环境变量指向）

## 学习路线

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。
