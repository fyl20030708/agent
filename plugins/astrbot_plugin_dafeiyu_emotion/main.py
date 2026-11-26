"""AstrBot 插件：普通版不看情绪，情绪版用本地 MacBERT。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from emotion.debug_clf import DebugClassifier
from emotion.engine import DEFAULT_SWITCH_KEYWORDS, Engine
from emotion.inject import attach_instruction
from emotion.macbert import MacBertClassifier

PERSONA_MARK = "【大肥鱼情绪版】回复时遵守本轮回复纸条，不要改判情绪。"


@register("astrbot_plugin_dafeiyu_emotion", "fyl20030708", "大肥鱼情绪版", "0.1.0")
class DaFeiYuEmotion(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self.engine: Engine | None = None
        self.load_error = ""

    async def initialize(self) -> None:
        try:
            self.engine = self._build_engine()
            self.engine.store_path = self._state_path()
            self.engine.load()
        except Exception as exc:  # noqa: BLE001
            self.load_error = str(exc)
            logger.error("大肥鱼情绪版还不能加载 MacBERT：%s", exc)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        text = (event.message_str or "").strip()
        command = _command(text)
        session_id = event.unified_msg_origin
        if command == "emotion":
            if self.engine is None:
                event.stop_event()
                yield event.plain_result(f"情绪版还没准备好：{self.load_error}")
                return
            self.engine.set_mode(session_id, "emotion")
            self.engine.save()
            event.stop_event()
            yield event.plain_result(
                "情绪版开了。每句话先用本机 MacBERT 看情绪，回复仍用你现在选的模型。"
                "发「普通版」就关掉。"
            )
            return
        if command == "normal":
            if self.engine is not None:
                self.engine.set_mode(session_id, "normal")
                self.engine.save()
            event.stop_event()
            yield event.plain_result("回到普通版。不再看情绪。")
            return
        if self.engine is None or self.engine.mode(session_id) != "emotion":
            if self.engine is None:
                return
            if not _bot_is_listening(event):
                return
            slip = self.engine.suggest_switch(session_id, text, datetime.now(timezone.utc))
            if slip is None:
                return
            direct = _apply_slip(self.engine, event, slip)
            if direct:
                yield event.plain_result(direct)
            return
        if not _bot_is_listening(event):
            return
        slip = self.engine.handle(session_id, text, datetime.now(timezone.utc))
        direct = _apply_slip(self.engine, event, slip)
        if direct:
            yield event.plain_result(direct)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req) -> None:
        instruction = event.get_extra("dafeiyu_slip")
        if not instruction:
            return
        system_prompt = getattr(req, "system_prompt", "") or ""
        if PERSONA_MARK not in system_prompt:
            req.system_prompt = f"{system_prompt}\n{PERSONA_MARK}".strip()
        attach_instruction(req, instruction)

    def _build_engine(self) -> Engine:
        kind = str(self.config.get("classifier") or "debug").strip()
        hours = float(self.config.get("min_hours") or 3)
        turns = int(self.config.get("min_turns") or 6)
        keywords = _switch_keywords(self.config.get("switch_keywords"))
        if kind == "debug":
            return Engine(DebugClassifier(), min_hours=hours, min_turns=turns, switch_keywords=keywords)
        classifier = MacBertClassifier(
            str(self.config.get("macbert_base_dir") or ""),
            str(self.config.get("macbert_checkpoint") or ""),
            str(self.config.get("macbert_device") or "cpu"),
        )
        return Engine(classifier, min_hours=hours, min_turns=turns, switch_keywords=keywords)

    def _state_path(self) -> Path:
        custom = str(self.config.get("state_file") or "").strip()
        if custom:
            return Path(custom)
        return Path("data") / "plugin_data" / "astrbot_plugin_dafeiyu_emotion" / "state.json"


def _bot_is_listening(event: AstrMessageEvent) -> bool:
    return bool(event.is_private_chat() or event.is_wake or event.is_at_or_wake_command)


def _apply_slip(engine: Engine, event: AstrMessageEvent, slip) -> str | None:
    engine.save()
    event.set_extra("dafeiyu_slip", slip.instruction)
    if not slip.call_model and slip.direct_reply:
        event.stop_event()
        return slip.direct_reply
    return None


def _switch_keywords(raw) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return DEFAULT_SWITCH_KEYWORDS
    parts = [item.strip() for item in text.replace("，", ",").split(",")]
    chosen = tuple(item for item in parts if item)
    return chosen or DEFAULT_SWITCH_KEYWORDS


def _command(text: str) -> str:
    normalized = text.strip().lstrip("/").strip()
    if normalized in {"情绪版", "情绪模式"}:
        return "emotion"
    if normalized in {"普通版", "普通模式"}:
        return "normal"
    return ""
