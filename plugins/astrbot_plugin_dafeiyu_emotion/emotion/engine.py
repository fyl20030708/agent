"""情绪版的判断和回复纸条。聊天模型只负责把纸条说出来。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LABELS = ("happy", "angry", "sad", "fear", "surprise", "neutral")
CARE_STATES = ("low", "angry", "stressed", "afraid")
ASK_AT = {"low": 0.75, "angry": 0.80, "stressed": 0.45, "afraid": 0.85}
QUESTIONS = {
    "low": "这几句听起来有点沉，是遇到事了，还是我多想了？",
    "angry": "这几句有点火气。是在生气，还是我听岔了？",
    "stressed": "听起来像是在扛事情，是压力上来了，还是我理解偏了？",
    "afraid": "这几句像是在怕什么。是真的怕，还是我多想了？",
}
STATE_NAME = {
    "low": "低落",
    "angry": "生气",
    "stressed": "有压力",
    "afraid": "害怕",
    "pleasant": "高兴",
    "calm": "平静",
    "unclear": "说不清",
    "crisis": "需要帮助",
}
MEDIA_CUES = ("电影", "剧情", "电视剧", "这集", "小说", "番剧")
STRESS_CUES = ("压力", "焦虑", "心慌", "睡不着", "忙不过来", "身心俱疲", "喘不过气", "扛不住")
FEAR_CUES = ("害怕", "好怕", "吓死", "心跳")
OPT_OUT = ("别问情绪", "不要问我情绪", "不要问情绪", "别关心我的情绪")
CRISIS = ("不想活", "活不下去", "自杀", "想自杀", "结束自己")
# 依据大连理工情感词汇本体的 7 类（乐、好、怒、哀、惧、恶、惊），
# 只保留聊天里会直接说出口、至少两个字的感受词。
# 不收入整本两万多条，避免「优秀」「奇怪」这类评价词也来询问。
DEFAULT_SWITCH_KEYWORDS = (
    "开心",
    "高兴",
    "快乐",
    "幸福",
    "兴奋",
    "喜悦",
    "愉快",
    "欢喜",
    "惊喜",
    "狂喜",
    "欣喜",
    "舒畅",
    "痛快",
    "得意",
    "自豪",
    "难过",
    "伤心",
    "悲伤",
    "难受",
    "痛苦",
    "委屈",
    "失落",
    "失望",
    "绝望",
    "崩溃",
    "郁闷",
    "孤独",
    "寂寞",
    "心累",
    "想哭",
    "心碎",
    "心寒",
    "心疼",
    "痛心",
    "悲哀",
    "悲痛",
    "凄凉",
    "落寞",
    "消沉",
    "压抑",
    "空虚",
    "厌世",
    "灰心",
    "不开心",
    "心情不好",
    "生气",
    "愤怒",
    "恼火",
    "火大",
    "窝火",
    "来气",
    "气愤",
    "暴怒",
    "发怒",
    "害怕",
    "恐惧",
    "心慌",
    "慌张",
    "焦虑",
    "担心",
    "紧张",
    "压力",
    "不安",
    "恐慌",
    "后怕",
    "睡不着",
    "失眠",
    "身心俱疲",
    "扛不住",
    "喘不过气",
    "忙不过来",
    "不知所措",
    "坐立不安",
    "烦躁",
    "心烦",
    "烦恼",
    "烦闷",
    "憋屈",
    "窝囊",
    "讨厌",
    "厌恶",
    "恶心",
    "反感",
    "好烦",
    "内疚",
    "愧疚",
    "后悔",
    "吃惊",
    "震惊",
    "吓到",
    "惊讶",
    "害羞",
    "羞愧",
    "丢脸",
    "想念",
    "牵挂",
)
SWITCH_DECLINE_HOURS = 24
YES = {"是", "是的", "对", "对的", "嗯", "有", "有点", "有一点", "不开心", "是啊"}
CRISIS_REPLY = (
    "我在。这种时候我先不往下分析，也不岔开话题。"
    "如果你正在想到伤害自己，可以打北京心理援助热线 010-82951332，"
    "或者当地的 24 小时心理热线。你愿意的话，我就在这儿听你说。"
)


@dataclass
class EmotionScore:
    pred: str
    probs: dict[str, float]
    confidence: float

    @classmethod
    def from_probs(cls, probs: dict[str, float]) -> EmotionScore:
        filled = {label: float(probs.get(label, 0.0)) for label in LABELS}
        total = sum(filled.values())
        if total <= 0:
            filled = {label: float(label == "neutral") for label in LABELS}
        else:
            filled = {label: value / total for label, value in filled.items()}
        top = max(range(len(LABELS)), key=lambda index: (filled[LABELS[index]], -index))
        return cls(LABELS[top], filled, filled[LABELS[top]])

    @classmethod
    def from_payload(cls, payload: dict) -> EmotionScore:
        if isinstance(payload, EmotionScore):
            return payload
        return cls.from_probs(payload.get("probs") or {})


@dataclass
class Slip:
    action_state: str
    call_model: bool
    instruction: str
    direct_reply: str | None = None
    ask: bool = False


@dataclass
class Memory:
    ask_emotion: bool = True
    pending: str | None = None
    pending_state: str | None = None
    pending_pred: str | None = None
    pending_confidence: float | None = None
    last_care_at: datetime | None = None
    last_switch_ask_at: datetime | None = None
    switch_declined_at: datetime | None = None
    turns_since_care: int = 1_000_000
    episodes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ask_emotion": self.ask_emotion,
            "pending": self.pending,
            "pending_state": self.pending_state,
            "pending_pred": self.pending_pred,
            "pending_confidence": self.pending_confidence,
            "last_care_at": self.last_care_at.isoformat() if self.last_care_at else None,
            "last_switch_ask_at": self.last_switch_ask_at.isoformat() if self.last_switch_ask_at else None,
            "switch_declined_at": self.switch_declined_at.isoformat() if self.switch_declined_at else None,
            "turns_since_care": self.turns_since_care,
            "episodes": self.episodes[-200:],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Memory:
        stamp = raw.get("last_care_at")
        asked = raw.get("last_switch_ask_at")
        declined = raw.get("switch_declined_at")
        return cls(
            ask_emotion=bool(raw.get("ask_emotion", True)),
            pending=raw.get("pending"),
            pending_state=raw.get("pending_state"),
            pending_pred=raw.get("pending_pred"),
            pending_confidence=raw.get("pending_confidence"),
            last_care_at=datetime.fromisoformat(stamp) if stamp else None,
            last_switch_ask_at=datetime.fromisoformat(asked) if asked else None,
            switch_declined_at=datetime.fromisoformat(declined) if declined else None,
            turns_since_care=int(raw.get("turns_since_care", 1_000_000)),
            episodes=list(raw.get("episodes") or []),
        )


class Engine:
    def __init__(
        self,
        classifier,
        min_hours: float = 3,
        min_turns: int = 6,
        switch_keywords: tuple[str, ...] = DEFAULT_SWITCH_KEYWORDS,
        switch_decline_hours: float = SWITCH_DECLINE_HOURS,
    ) -> None:
        self.classifier = classifier
        self.min_hours = min_hours
        self.min_turns = min_turns
        self.switch_keywords = switch_keywords
        self.switch_decline_hours = switch_decline_hours
        self.memories: dict[str, Memory] = {}
        self.modes: dict[str, str] = {}
        self.store_path: Path | None = None

    def memory(self, session_id: str) -> Memory:
        if session_id not in self.memories:
            self.memories[session_id] = Memory()
        return self.memories[session_id]

    def mode(self, session_id: str) -> str:
        return self.modes.get(session_id, "normal")

    def set_mode(self, session_id: str, mode: str) -> None:
        self.modes[session_id] = mode
        self.memory(session_id).pending = None

    def suggest_switch(self, session_id: str, text: str, now: datetime) -> Slip | None:
        """普通版里遇到关心词时，问一次要不要换成情绪版。"""
        if self.mode(session_id) != "normal":
            return None
        memory = self.memory(session_id)
        text = (text or "").strip()
        if _contains_any(text, CRISIS):
            memory.pending = None
            return Slip("crisis", False, "", CRISIS_REPLY, False)
        if memory.pending == "switch":
            memory.pending = None
            if _switch_decline(text):
                memory.switch_declined_at = now
                return _tone("calm", "对方选择继续普通聊天。不要再提情绪版，按普通聊天接话。")
            return None
        if _switch_on_cooldown(memory, now, self.min_hours, self.switch_decline_hours):
            return None
        if not _contains_any(text, self.switch_keywords):
            return None
        memory.pending = "switch"
        memory.last_switch_ask_at = now
        return Slip(
            "calm",
            True,
            _slip(
                [
                    "对方还在普通版，这句话里出现了需要关心的说法。",
                    "先接住他刚才的事，然后只问一次，不要改写：",
                    "「要不要换成情绪版？回情绪版就换，回不用就继续这样聊。」",
                    "不要开始分析情绪，不要讲故事。",
                ]
            ),
        )

    def handle(self, session_id: str, text: str, now: datetime) -> Slip:
        memory = self.memory(session_id)
        text = (text or "").strip()
        if _contains_any(text, CRISIS):
            memory.pending = None
            return Slip("crisis", False, "", CRISIS_REPLY, False)
        if _contains_any(text, OPT_OUT):
            memory.ask_emotion = False
            memory.pending = None
            return _tone("unclear", "主人说过不要问情绪。这轮按普通聊天接话。")

        if memory.pending == "confirm":
            kind = _reply_kind(text)
            if kind == "yes":
                memory.pending = "choice"
                return _offer(memory.pending_state or "low")
            if kind == "no":
                memory.episodes.append(
                    {
                        "text": text,
                        "pred": memory.pending_pred,
                        "confirmed_state": "none",
                        "intervention": "ask",
                    }
                )
                memory.pending = None
                memory.turns_since_care += 1
                return _tone("unclear", "对方否认了刚才的情绪判断。按普通聊天接他现在的话。")
            memory.pending = None

        if memory.pending == "choice":
            choice = _parse_choice(text, memory.pending_state or "")
            if choice:
                state = memory.pending_state or "low"
                memory.episodes.append(
                    {
                        "text": text,
                        "pred": memory.pending_pred,
                        "confirmed_state": state,
                        "intervention": choice,
                    }
                )
                memory.pending = None
                memory.turns_since_care += 1
                return _after_choice(state, choice)
            memory.pending = None

        memory.turns_since_care += 1
        if _contentless(text):
            return _tone("calm", "这句没有可判断的文字。按普通聊天接话。")
        if _contains_any(text, MEDIA_CUES):
            self._predict(text)
            return _tone("unclear", "这句像是在说电影或剧情，不要当成主人此刻的心情。")

        score = self._predict(text)
        state = _decide_state(text, score)
        if state in ASK_AT and score.confidence >= ASK_AT[state] and _care_allowed(memory, now, self.min_hours, self.min_turns):
            memory.pending = "confirm"
            memory.pending_state = state
            memory.pending_pred = score.pred
            memory.pending_confidence = score.confidence
            memory.last_care_at = now
            memory.turns_since_care = 0
            return _ask(state, score)
        return _tone(state, "这句不要问情绪，顺着对方的内容聊。")

    def save(self) -> None:
        if self.store_path is None:
            return
        payload = {
            "modes": self.modes,
            "sessions": {key: value.to_dict() for key, value in self.memories.items()},
        }
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if self.store_path is None or not self.store_path.is_file():
            return
        raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.modes = dict(raw.get("modes") or {})
        self.memories = {key: Memory.from_dict(value) for key, value in (raw.get("sessions") or {}).items()}

    def _predict(self, text: str) -> EmotionScore:
        score = self.classifier.predict(text)
        if isinstance(score, EmotionScore):
            return score
        return EmotionScore.from_payload(score)


def _switch_decline(text: str) -> bool:
    stripped = text.strip()
    if stripped in {"不用", "不要", "不用了", "不要换", "算了", "继续", "继续普通"}:
        return True
    return stripped.startswith(("不用", "不要换", "继续普通"))


def _switch_on_cooldown(memory: Memory, now: datetime, min_hours: float, decline_hours: float) -> bool:
    if memory.switch_declined_at is not None:
        declined_hours = (now - memory.switch_declined_at).total_seconds() / 3600
        if declined_hours < decline_hours:
            return True
    if memory.last_switch_ask_at is None:
        return False
    asked_hours = (now - memory.last_switch_ask_at).total_seconds() / 3600
    return asked_hours < min_hours


def _care_allowed(memory: Memory, now: datetime, min_hours: float, min_turns: int) -> bool:
    if not memory.ask_emotion:
        return False
    if memory.last_care_at is None:
        return True
    hours = (now - memory.last_care_at).total_seconds() / 3600
    return hours >= min_hours or memory.turns_since_care >= min_turns


def _decide_state(text: str, score: EmotionScore) -> str:
    if _contains_any(text, STRESS_CUES) and not (score.pred == "happy" and score.confidence >= 0.7):
        negative = score.probs.get("sad", 0) + score.probs.get("angry", 0) + score.probs.get("fear", 0)
        if score.pred in {"sad", "angry", "fear"} and score.confidence >= 0.45:
            return "stressed"
        if negative >= 0.45 and score.pred != "happy":
            return "stressed"
    if score.pred == "happy" and score.confidence >= 0.7:
        return "pleasant"
    if score.pred == "surprise":
        return "unclear"
    if score.pred == "neutral":
        return "calm"
    if score.pred == "sad":
        return "low"
    if score.pred == "angry":
        return "angry"
    if score.pred == "fear" and _contains_any(text, FEAR_CUES):
        return "afraid"
    return "unclear"


def _ask(state: str, score: EmotionScore) -> Slip:
    name = STATE_NAME[state]
    lines = [
        f"情绪判断：{name}（{score.pred}，置信度 {score.confidence:.2f}）。这是 MacBERT 的结果，不要改判。",
        "先用一两句接住对方刚才说的事。",
        f"然后只问这一句，不要改写：「{QUESTIONS[state]}」",
        "不要讲故事，不要给建议，问完停下。",
    ]
    return Slip(state, True, _slip(lines), ask=True)


def _offer(state: str) -> Slip:
    if state == "low":
        body = "请他选一个：听你说、讲个短故事，或者就这样待着。等他选，不要直接讲故事，不要给建议。"
    elif state == "stressed":
        body = "问他要不要把眼前的事理一下，或只是听着。不要讲故事。"
    else:
        body = "告诉他你听着，他可以接着说。不要给建议。"
    return Slip(state, True, _slip([f"对方确认自己{STATE_NAME.get(state, '有情绪')}。不要改判。", body]), ask=False)


def _after_choice(state: str, choice: str) -> Slip:
    if choice == "story":
        text = "对方确认自己有点低落，并选择听一个短故事。讲一个很短的故事陪他，讲完停下。不要分析情绪，不要说教。"
    elif choice == "clarify":
        text = "对方选择把事情理一下。不要改判，不要讲故事。只帮他说清眼前这一件最压着的事，不要展开成建议清单。"
    elif choice == "stay":
        text = "对方希望就这样待着。不要再问，轻轻接一句。不要讲故事。"
    else:
        text = "对方选择让你听着。不要改判。接一句，让他继续说。不要给建议，不要讲故事。"
    return Slip(state, True, _slip([text]), ask=False)


def _tone(state: str, sentence: str) -> Slip:
    return Slip(state, True, _slip([sentence]), ask=False)


def _slip(lines: list[str]) -> str:
    if not any("不要改判" in line or "不要分析情绪" in line for line in lines):
        lines = [*lines, "不要改判。"]
    return "【本轮回复纸条】\n" + "\n".join(lines)


def _reply_kind(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(("没有", "不是")) or "多想了" in stripped or "讲电影" in stripped or "开玩笑" in stripped:
        return "no"
    if stripped in YES:
        return "yes"
    return "other"


def _parse_choice(text: str, state: str) -> str | None:
    if "故事" in text and state == "low":
        return "story"
    if "理" in text and state == "stressed":
        return "clarify"
    if "就这样" in text or stripped_stay(text):
        return "stay"
    if "听" in text:
        return "listen"
    return None


def stripped_stay(text: str) -> bool:
    return text.strip() in {"不用", "算了", "待着", "就这样"}


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _contentless(text: str) -> bool:
    return not any("\u4e00" <= char <= "\u9fff" or char.isalnum() for char in text)
