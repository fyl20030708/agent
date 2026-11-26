"""把本轮回复纸条附到聊天请求上，不改长期人设。"""


def attach_instruction(req, instruction: str) -> None:
    """附上只对这一轮有效的纸条。人设系统提示词保持原样。"""
    parts = getattr(req, "extra_user_content_parts", None)
    if isinstance(parts, list):
        parts.append(_content_part(instruction))
        return
    current = getattr(req, "prompt", "") or ""
    req.prompt = f"{current}\n\n{instruction}"


def _content_part(instruction: str):
    try:
        from astrbot.core.agent.message import TextPart
    except ImportError:
        return {"type": "text", "text": instruction, "temporary": True}
    part = TextPart(text=instruction)
    mark = getattr(part, "mark_as_temp", None)
    if callable(mark):
        part = mark()
    return part
