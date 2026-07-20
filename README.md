# Agent 开发学习项目

从零学习 LLM Agent 工程，最终目标是做一个能读懂 `31P_transformer` 科研代码与实验数据的助手。

## 目录结构

```
agent/
├── docs/ROADMAP.md          # 12 周学习计划
├── src/zero_week/           # 阶段 0：async → 迷你 Agent 引擎
├── docs/superpowers/specs/  # 设计文档
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

## 关联项目

- 科研代码库：`/homes/yli/31P_transformer`（通过 `REPO_ROOT` 环境变量指向）

## 学习路线

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。
