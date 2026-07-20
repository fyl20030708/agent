"""
Async → Agent 引擎 学习路线（对应 zero.ipynb 全本）

Level 1  await / async def
Level 2  create_task（后台任务）
Level 3  gather（并发）
Level 4  Semaphore + Queue（限流 / 生产者消费者）
Level 5  conversation_loop + Budget + run_async 桥

在 Jupyter 里可按函数单独调用；也可直接：
    python async_agent_roadmap.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field


# =============================================================================
# Level 1 —— async def / await
# 直觉：协程是「可暂停的函数」；await 是暂停点，让出 event loop 去跑别人
# =============================================================================

async def L1_fetch() -> str:
    await asyncio.sleep(0.1)  # 模拟 I/O；CPU 不做忙等，控制权交回 loop
    return "hello"


async def L1_main() -> None:
    # 调用 L1_fetch() 只得到 coroutine 对象，不会立刻跑完
    # 必须 await（或 create_task / gather）才会真正执行
    print("[L1]", await L1_fetch())


# =============================================================================
# Level 2 —— create_task：主流程继续，副任务后台跑
# =============================================================================

async def L2_heartbeat() -> None:
    for i in range(3):
        print(f"  [heartbeat] tick {i}")
        await asyncio.sleep(0.4)


async def L2_fetch_data() -> dict:
    print("  [fetch] start")
    await asyncio.sleep(0.8)
    print("  [fetch] done")
    return {"data": 42}


async def L2_main() -> None:
    # create_task：立刻注册到 loop，不阻塞当前协程
    hb = asyncio.create_task(L2_heartbeat())
    data = await L2_fetch_data()
    print("[L2] main got:", data)
    await hb  # 收尾：别让后台任务被垃圾回收悄悄取消


# =============================================================================
# Level 3 —— gather：多个 I/O 同时等，总耗时 ≈ max(delay)
# =============================================================================

async def L3_call_tool(name: str, delay: float) -> str:
    print(f"  [{name}] start")
    await asyncio.sleep(delay)
    print(f"  [{name}] done")
    return f"{name}_result"


async def L3_main() -> None:
    t0 = time.perf_counter()
    results = await asyncio.gather(
        L3_call_tool("web_search", 1.0),
        L3_call_tool("read_file", 0.5),
        L3_call_tool("run_code", 0.8),
    )
    # return_exceptions=True 时，某个失败不会打爆整批（Agent tool 必需）
    print("[L3] results:", results)
    print(f"[L3] elapsed: {time.perf_counter() - t0:.2f}s  (≈1.0 not 2.3)")


# =============================================================================
# Level 4a —— Semaphore：同时最多 N 个在跑（API rate limit / 连接数）
# =============================================================================

async def L4a_main(max_concurrent: int = 2, n_jobs: int = 6) -> None:
    sem = asyncio.Semaphore(max_concurrent)

    async def one(job_id: int) -> None:
        async with sem:  # 进门拿票，出来还票
            print(f"  job-{job_id} running (slots≤{max_concurrent})")
            await asyncio.sleep(0.4)
            print(f"  job-{job_id} done")

    # 注意：必须 await gather，否则主协程立刻返回，任务可能还没跑完就被取消
    await asyncio.gather(*[one(i) for i in range(n_jobs)])
    print("[L4a] all jobs done")


# =============================================================================
# Level 4b —— Queue + Producer/Consumer（batch_runner / 任务池）
# =============================================================================

@dataclass
class Job:
    id: int
    prompt: str


async def L4b_main(n_workers: int = 3, n_jobs: int = 8, max_llm: int = 2) -> None:
    queue: asyncio.Queue[Job | None] = asyncio.Queue(maxsize=4)
    sem = asyncio.Semaphore(max_llm)

    async def producer(jobs: list[Job]) -> None:
        for j in jobs:
            await queue.put(j)
            print(f"[producer] enqueue job-{j.id}")
        # poison pill：每个 worker 收一个 None 后退出
        for _ in range(n_workers):
            await queue.put(None)

    async def worker(wid: int) -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    break
                async with sem:
                    print(f"[worker-{wid}] run job-{item.id}: {item.prompt!r}")
                    await asyncio.sleep(0.5)
                    print(f"[worker-{wid}] done job-{item.id}")
            finally:
                queue.task_done()

    jobs = [Job(i, f"prompt-{i}") for i in range(n_jobs)]
    await asyncio.gather(producer(jobs), *[worker(i) for i in range(n_workers)])
    print("[L4b] queue drained")


# =============================================================================
# Level 5a —— conversation_loop 骨架 + IterationBudget（consume）
# =============================================================================

@dataclass
class Message:
    role: str  # user | assistant | tool
    content: str


@dataclass
class IterationBudget:
    """简化版 Hermes IterationBudget：油表，不是普通计数器。"""

    max_total: int = 5
    used: int = 0

    def consume(self) -> bool:
        if self.used >= self.max_total:
            return False
        self.used += 1
        return True

    def refund(self) -> None:
        if self.used > 0:
            self.used -= 1


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
    budget: IterationBudget = field(default_factory=IterationBudget)


async def call_llm(messages: list[Message]) -> dict:
    await asyncio.sleep(0.2)
    last = messages[-1]
    # 关键：看 role，不要只搜 content 里的 "search"（否则 tool result 会永远触发 tool_call）
    if last.role == "user" and "search" in last.content.lower():
        return {"type": "tool_call", "tool": "web_search", "args": {"q": last.content}}
    return {"type": "text", "content": f"Reply after tools; last={last.content[:40]!r}"}


async def execute_tool_async(name: str, args: dict) -> str:
    await asyncio.sleep(0.3)
    return f"[{name}]({args}) => mock result"


async def run_turn(state: AgentState, user_msg: str) -> str:
    """≈ Hermes conversation_loop：一轮 user turn 的调度器。"""
    state.messages.append(Message("user", user_msg))

    while True:
        if not state.budget.consume():
            return "[halted] budget exhausted"

        print(f"\n── iteration {state.budget.used}/{state.budget.max_total} ──")
        response = await call_llm(state.messages)

        if response["type"] == "text":
            state.messages.append(Message("assistant", response["content"]))
            return response["content"]

        if response["type"] == "tool_call":
            # 这里先用裸 await；下面 Level 5b 会改成 sync + run_async
            result = await execute_tool_async(response["tool"], response["args"])
            state.messages.append(Message("tool", result))
            # 不 return：把 tool 结果塞回 history，下一轮再 call_llm


async def L5a_main() -> None:
    state = AgentState(budget=IterationBudget(max_total=5))
    answer = await run_turn(state, "please search python async")
    print(f"\n[L5a] Final: {answer}")
    print(f"[L5a] used: {state.budget.used}")


# =============================================================================
# Level 5b —— _run_async：sync 世界调用 async tool（Hermes model_tools 模式）
# =============================================================================

_tool_loop: asyncio.AbstractEventLoop | None = None
_worker_local = threading.local()


def _get_tool_loop() -> asyncio.AbstractEventLoop:
    global _tool_loop
    if _tool_loop is None or _tool_loop.is_closed():
        _tool_loop = asyncio.new_event_loop()
    return _tool_loop


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_worker_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_local.loop = loop
    return loop


def run_async(coro):
    """
    简化版 Hermes `_run_async`：从 sync 上下文跑一个 coroutine。

    三条路径：
    1. 已有 running loop（gateway）→ 丢到新线程，避免 "loop already running"
    2. worker 线程 → 线程级持久 loop（缓存的 httpx 客户端不会绑到已关闭的 loop）
    3. 主线程 CLI → 进程级持久 loop
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None and running.is_running():
        def _worker():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_worker).result(timeout=30)

    if threading.current_thread() is not threading.main_thread():
        return _get_worker_loop().run_until_complete(coro)

    return _get_tool_loop().run_until_complete(coro)


def handle_function_call(name: str, args: dict) -> str:
    """≈ Hermes handle_function_call：对外是 sync API。"""
    if name == "web_search":
        return run_async(execute_tool_async(name, args))
    raise ValueError(f"unknown tool: {name}")


def L5b_main() -> None:
    # 故意用 sync 入口，演示「没有 await 的世界」怎么跑 async tool
    print("[L5b]", handle_function_call("web_search", {"q": "hermes agent"}))


# =============================================================================
# Level 5c —— 合成引擎：loop + gather + Semaphore + Budget
# =============================================================================

@dataclass
class AgentEngine:
    max_iterations: int = 5
    max_concurrent_tools: int = 2
    messages: list[dict] = field(default_factory=list)
    _sem: asyncio.Semaphore = field(init=False)
    budget: IterationBudget = field(init=False)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_concurrent_tools)
        self.budget = IterationBudget(max_total=self.max_iterations)

    async def llm_call(self) -> dict:
        await asyncio.sleep(0.15)
        # 只有第一条 user、还没跑过 tool 时，才要 3 个并行 tool
        if len(self.messages) == 1:
            return {
                "tool_calls": [
                    {"name": "search", "args": {"q": "a"}},
                    {"name": "search", "args": {"q": "b"}},
                    {"name": "read", "args": {"path": "/tmp/x"}},
                ]
            }
        return {"content": "Here is your answer."}

    async def _run_one_tool(self, call: dict) -> dict:
        async with self._sem:
            name, args = call["name"], call["args"]
            print(f"    tool {name}({args}) start")
            await asyncio.sleep(0.4)
            print(f"    tool {name}({args}) done")
            return {"tool": name, "result": f"mock-{name}"}

    async def dispatch_tools(self, tool_calls: list[dict]) -> list[dict]:
        tasks = [asyncio.create_task(self._run_one_tool(tc)) for tc in tool_calls]
        return await asyncio.gather(*tasks)

    async def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        while True:
            if not self.budget.consume():
                return "[budget exhausted]"
            print(f"\n=== Turn iteration {self.budget.used} ===")
            resp = await self.llm_call()
            if "content" in resp:
                return resp["content"]
            if "tool_calls" in resp:
                results = await self.dispatch_tools(resp["tool_calls"])
                self.messages.append({"role": "tools", "content": results})


async def L5c_main() -> None:
    engine = AgentEngine(max_iterations=5, max_concurrent_tools=2)
    answer = await engine.run("analyze this repo")
    print(f"\n[L5c] Answer: {answer}")


# =============================================================================
# 入口：按顺序跑一遍全地图
# =============================================================================

async def demo_all_async() -> None:
    print("\n======== Level 1 ========")
    await L1_main()
    print("\n======== Level 2 ========")
    await L2_main()
    print("\n======== Level 3 ========")
    await L3_main()
    print("\n======== Level 4a ========")
    await L4a_main()
    print("\n======== Level 4b ========")
    await L4b_main()
    print("\n======== Level 5a ========")
    await L5a_main()
    print("\n======== Level 5c ========")
    await L5c_main()


if __name__ == "__main__":
    asyncio.run(demo_all_async())
    print("\n======== Level 5b (sync bridge) ========")
    L5b_main()
