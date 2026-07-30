# Minimal RAG（week04）

## 流水线

```text
REPO_ROOT/**/*.md
    → discover + chunk (path, heading, text, mtime)
    → embed  POST {OPENAI_BASE_URL}/embeddings
    → cache  week04/.rag_cache.json
    → retrieve  query embed → cosine top-k
    → answer  LLM + CONTEXT prompt + path#heading 引用
```

| 阶段 | 代码 | 做什么 |
|------|------|--------|
| **chunk** | `rag_lib.chunk_markdown` | 按 `#`–`###` 标题切分；无标题则 ~800 字窗口 + 100 字重叠 |
| **embed** | `rag_lib.embed_texts` | 批量 POST `/embeddings`，每块一个向量 |
| **cache** | `rag_lib.build_index` | 读/写 JSON；mtime 未变则复用向量 |
| **retrieve** | `rag_lib.search` | 查询向量 vs 全库 cosine，取 top-k |
| **answer** | `rag_demo._chat_answer` | 把命中块塞进 CONTEXT，要求模型只依据 CONTEXT 作答并引用 `path#heading` |

纯函数与 IO 在 `rag_lib.py`；CLI 在 `rag_demo.py`。`week04` **不** import `research_agent`。

## 工具怎么选

迁入 `research_agent` 后，`search_docs` 与 week02 的 `read_file` / `grep_repo` 并列。原则：**检索语义，不替代精确读文件。**

| 需求 | 用 | 原因 |
|------|-----|------|
| 概念 / 设计 / “X 是怎么工作的” | **`search_docs`** | 跨多篇 `.md` 的语义相似度；不必先知道文件名 |
| 某文件的**确切**内容、行级核对 | **`read_file`** | 全文或指定片段；无 embedding 近似误差 |
| 代码里**精确**符号、字符串、函数名 | **`grep_repo`** | 正则/字面匹配 `.py`；RAG 本周只索引 `*.md` |
| 目录结构、文件是否存在 | **`list_dir`** | 与 RAG 无关的导航 |

**一句话：** RAG 帮你**找相关段落**；`read_file` 帮你**读对文件**；`grep_repo` 帮你**定位符号**。

### 何时**不要**用 RAG

- 已知路径，只需打开那一个文件 → `read_file`
- 找 `class Foo`、`def bar(`、配置键名 → `grep_repo`
- 问题关于**源码实现**而答案只在 `.py` 里（本周 corpus 不含 `.py`）
- 需要**逐字**引用（许可证、数值、命令行 flags）→ 检索后再 `read_file` 核对
- 语料库外的事实（联网、其它 repo）→ RAG 给不了；应明说“文档未覆盖”
- 嵌入 API 不可用（见下）→ 先修 env，别假装有检索

## 缓存

**路径：** `week04/.rag_cache.json`（**gitignored**，勿提交）

**内容（每条 record）：**

```json
{
  "path": "Training/MAGNITUDE_TRAINING_GUIDE.md",
  "heading": "Overview",
  "text": "...",
  "mtime": 1710000000.0,
  "vector": [0.01, -0.02, ...]
}
```

顶层还有 `chunk_version`（当前 `1`）。版本 bump 时整库重 embed。

**mtime 失效：** `build_index` 对每个 `.md` 比较磁盘 `st_mtime` 与缓存里该 path 的 `mtime`。相同 → 复用该文件全部 chunk+vector；不同 → 重新 chunk 并 embed。**编辑文档后**下次运行会自动更新对应文件，无需手删 cache（除非 chunk 算法/`chunk_version` 变了）。

**为何 gitignore：** 向量体积大、随本地语料变化、可本地重建；提交会污染 diff 且可能泄露语料快照。

## 环境变量

| 变量 | 作用 | 默认 |
|------|------|------|
| `OPENAI_API_KEY` | 鉴权 | 必填 |
| `OPENAI_BASE_URL` | API 根 | `https://api.deepseek.com/v1` |
| `OPENAI_MODEL` | 回答用的 chat 模型 | `deepseek-chat` |
| **`OPENAI_EMBED_MODEL`** | 嵌入模型 | **`ecnu-embedding-small`** |
| `REPO_ROOT` | 索引的 Markdown 根目录 | `/homes/yli/31P_transformer` |

### DeepSeek 与 `/embeddings`

**DeepSeek 聊天 API 没有 `/embeddings` 端点。** 若 `OPENAI_BASE_URL` 指向 DeepSeek，embed 请求会失败。

可行做法：

1. **推荐（本仓库验证路径）：** chat 与 embed 都用 ECNU — 在 `.env` 里设 `OPENAI_BASE_URL` 为 ECNU 网关，并设 `OPENAI_EMBED_MODEL=ecnu-embedding-small`。
2. 仅 chat 用 DeepSeek、embed 用 ECNU — 需要两个 base URL；**本周代码共用一个 `OPENAI_BASE_URL`**，因此 Learner 应把 base 指到**提供 `/embeddings` 的**提供商，或接受 chat+embed 同域（ECNU）。

Chat 与 embed **可以**用不同厂商；本周 demo **不能**在 DeepSeek-only 的 base 上完成首次建索引。

## 与 ReAct 的 grounding 关系

week02 ReAct：`Action` 执行工具 → 环境返回 **`Observation`（事实）** → 模型在下一轮基于 Observation 推理，**不能编造**工具没返回的内容。

RAG 同一规则：

| ReAct | RAG / `search_docs` |
|-------|---------------------|
| `Observation: ...` | 检索命中列表 + snippet（score、path#heading、text） |
| 模型不得虚构 Observation | 模型不得虚构**未出现在 CONTEXT / 工具返回里的 path 或段落** |
| `finish` 应 grounded | 最终答案必须引用 CONTEXT 里真实存在的 `path#heading`；覆盖不足则明说 |

`rag_demo` 的 system prompt 已写：**“Answer using only the supplied CONTEXT … Do not invent paths … If not covered, say so.”** 迁入 agent 后，`search_docs` 的 Observation 字符串扮演同一角色。

## 常用命令

以下命令在 **`week04/` 目录下**执行（`rag_demo.py` 通过 `import rag_lib` 加载同目录模块；等价做法：在仓库根目录设 `PYTHONPATH=week04` 后运行 `python week04/rag_demo.py …`）：

```bash
cd week04

# 首次运行：discover → chunk → embed → 写 cache；打印 top-k + LLM 答案
python rag_demo.py "How is magnitude training configured?"

# 只检索，不调 chat（看 score 与 path#heading）
python rag_demo.py "How is magnitude training configured?" --search-only

# 改 top-k（默认 5）
python rag_demo.py "magnitude training strategy" -k 3 --search-only
```

**第二次**对同一语料运行：日志应出现 `[week04] reused document cache`，不再 bulk embed。

迁入 `research_agent` 后（Task 6），同一检索逻辑以工具形式调用：

```bash
AGENT_VERBOSE=1 python -m research_agent.cli "Explain the magnitude training strategy and cite the doc."
```

verbose 日志里应出现 `search_docs` 一轮；最终答案仍须满足路径引用契约。

## 和后续 week 的关系

- **week04 demo：** 自包含 CLI，练 chunk/embed/cache/retrieve/answer 全链路。
- **research_agent：** 薄封装 `search_docs`，复用 `rag_lib` + 同一 cache；system prompt 补充与 `read_file` / `grep_repo` 的分工。
- **非目标（本周）：** 向量数据库、BM25+重排、索引 `.py`、多跳 Agentic RAG。
