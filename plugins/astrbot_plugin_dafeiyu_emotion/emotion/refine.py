"""聊天模型只把 MacBERT 的「悲伤」说细，不改六类粗分类。"""

FINE_LABELS = ("有压力", "委屈", "低落", "说不清")


def refinement_prompt(text: str, history: list[str]) -> str:
    earlier = "\n".join(f"- {line}" for line in history[-4:]) or "（没有更早的原话）"
    return (
        "MacBERT 已把当前这句判为悲伤。它没有低落、有压力、委屈这些类。\n"
        "请只根据原话，从「低落」「有压力」「委屈」「说不清」里选一个。\n"
        "不要改成高兴、愤怒、恐惧、惊奇或无情绪。只输出四个词中的一个。\n"
        f"更早的原话：\n{earlier}\n"
        f"当前这句：{text}"
    )


def parse_fine_label(raw: str) -> str:
    for label in FINE_LABELS:
        if label in (raw or ""):
            return label
    return "说不清"


def debug_fine_label(text: str) -> str:
    """没有聊天模型时，用原话里的词暂代细判断。"""
    if any(word in text for word in ("压力", "焦虑", "睡不着", "身心俱疲", "忙不过来", "扛不住", "喘不过气")):
        return "有压力"
    if "委屈" in text:
        return "委屈"
    return "低落"
