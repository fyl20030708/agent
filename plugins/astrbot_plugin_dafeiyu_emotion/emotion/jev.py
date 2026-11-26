"""JEV：模型只列出这一轮可以做的选择。程序收下这张表，再请模型按表和原话回复。"""

import json
import re

from emotion.engine import DECIDE_LINE


def jev_request(facts: str) -> str:
    source = facts.replace(DECIDE_LINE, "").replace("不要改判。", "").strip()
    return (
        f"{source}\n"
        "只输出一个 JSON 对象，不要写给主人的回复。\n"
        "choices 里放 2 到 3 个选择。do 是这一步做什么，不是对主人说的话。"
        "because 必须点到粗分类或原话里的词。\n"
        '{"choices":[{"do":"做什么","because":"依据"}]}'
    )


def parse_jev(raw: str) -> list[dict[str, str]]:
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    choices = []
    for item in payload.get("choices") or []:
        if not isinstance(item, dict):
            continue
        do = str(item.get("do") or "").strip()
        because = str(item.get("because") or "").strip()
        if do and because:
            choices.append({"do": do, "because": because})
    return choices[:3]


def format_jev(choices: list[dict[str, str]]) -> str:
    lines = ["JEV 选择："]
    for index, choice in enumerate(choices, start=1):
        lines.append(f"{index}. {choice['do']}。依据：{choice['because']}")
    return "\n".join(lines)


def apply_jev(instruction: str, choices: list[dict[str, str]]) -> str:
    if not choices:
        return instruction
    replacement = (
        f"{format_jev(choices)}\n"
        "根据上面的 JEV 选择和原话回复。一轮只用其中一个选择。不要改粗分类。"
        "情绪的主语用主人，沿用原话的说法。原话是厌恶三花，就写主人厌恶三花，不要写成厌恶对着三花。"
    )
    if DECIDE_LINE in instruction:
        return instruction.replace(DECIDE_LINE, replacement)
    return f"{instruction}\n{replacement}"
