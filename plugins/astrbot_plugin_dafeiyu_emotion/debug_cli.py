"""在本机看回复纸条，不启动 QQ。默认用关键词分类器。"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emotion.debug_clf import DebugClassifier
from emotion.engine import Engine


def main() -> None:
    bot = Engine(DebugClassifier())
    print("情绪版联调。这里打印给聊天模型的回复纸条，不调用模型。")
    print("有 MacBERT 权重后，在 AstrBot 插件配置里把 classifier 改成 macbert。")
    print("退出输入 quit")
    while True:
        try:
            text = input("你> ").strip()
        except EOFError:
            return
        if text in {"quit", "exit", "q"}:
            return
        if not text:
            continue
        slip = bot.handle("local", text, datetime.now(timezone.utc))
        if not slip.call_model:
            print("直接回复>")
            print(slip.direct_reply)
        else:
            print(slip.instruction)
        print()


if __name__ == "__main__":
    main()
