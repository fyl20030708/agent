import asyncio

# 协程函数：调用后返回 coroutine 对象，不会立刻执行
async def fetch():
    await asyncio.sleep(5)  # 模拟 I/O，让出控制权
    return "hello"

async def main():
    # await = "等这个 I/O 完成"
    result = await fetch()
    print(result)

asyncio.run(main())