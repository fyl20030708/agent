"""情绪版的判断和回复纸条。聊天模型只负责把纸条说出来。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LABELS = ("happy", "angry", "sad", "fear", "surprise", "neutral")
CARE_STATES = ("low", "angry", "stressed", "afraid")
# 愤怒门槛按实测下调：三花那句愤怒 66.9% 时仍要进入情绪回复。
ASK_AT = {"low": 0.75, "angry": 0.60, "stressed": 0.45, "afraid": 0.85, "wronged": 0.75}
STATE_NAME = {
    "low": "低落",
    "angry": "生气",
    "stressed": "有压力",
    "afraid": "害怕",
    "wronged": "委屈",
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
    recent_user: list = field(default_factory=list)
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
            "recent_user": self.recent_user[-6:],
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
            recent_user=list(raw.get("recent_user") or []),
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

    def handle(
        self,
        session_id: str,
        text: str,
        now: datetime,
        fine_label: str | None = None,
        known_score: EmotionScore | None = None,
    ) -> Slip:
        memory = self.memory(session_id)
        text = (text or "").strip()
        if _contains_any(text, CRISIS):
            memory.pending = None
            return Slip("crisis", False, "", CRISIS_REPLY, False)
        if _contains_any(text, OPT_OUT):
            memory.ask_emotion = False
            memory.pending = None
            return _tone("unclear", "主人说过不要问情绪。这轮按普通聊天接话。")

        _remember(memory, text)
        if memory.pending == "care":
            if _reply_kind(text) == "no":
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
            if _wants_story(text) and memory.pending_state in {"low", "wronged"}:
                return self._locked_action(memory, "story", now, log=True)
            if _wants_listen(text):
                return self._locked_action(memory, "listen", now, log=False)
            if "理一下" in text:
                return self._locked_action(memory, "unpack", now, log=True)
            if _wants_stop(text):
                return self._locked_action(memory, "stay", now, log=False)
        memory.turns_since_care += 1
        if _contentless(text):
            return _tone("calm", "这句没有可判断的文字。按普通聊天接话。")
        if _contains_any(text, MEDIA_CUES):
            if known_score is None:
                self._predict(text)
            memory.pending = None
            return _tone("unclear", "这句像是在说电影或剧情，不要当成主人此刻的心情。")

        score = known_score if known_score is not None else self._predict(text)
        state = _decide_state(text, score, fine_label)
        if memory.pending == "care" and state in ASK_AT:
            return self._begin_care(memory, state, score, now, reset_cooldown=False)
        if state in ASK_AT and score.confidence >= ASK_AT[state] and _care_allowed(memory, now, self.min_hours, self.min_turns):
            return self._begin_care(memory, state, score, now, reset_cooldown=True)
        memory.pending = None
        return _tone(state, "这句不要问情绪，顺着对方的内容聊。")

    def _begin_care(self, memory: Memory, state: str, score: EmotionScore, now: datetime, reset_cooldown: bool) -> Slip:
        memory.pending = "care"
        memory.pending_state = state
        memory.pending_pred = score.pred
        memory.pending_confidence = score.confidence
        if reset_cooldown:
            memory.last_care_at = now
            memory.turns_since_care = 0
        explicit = _explicit_action(state, memory.recent_user[-1] if memory.recent_user else "")
        if explicit:
            return _action_slip(state, score, explicit, memory.recent_user, ask=reset_cooldown)
        return _choice_slip(state, score, memory.recent_user, ask=reset_cooldown)

    def _locked_action(self, memory: Memory, action: str, now: datetime, log: bool) -> Slip:
        state = memory.pending_state or "low"
        score = EmotionScore(memory.pending_pred or "sad", {}, memory.pending_confidence or 0.0)
        if log:
            memory.episodes.append(
                {
                    "text": memory.recent_user[-1] if memory.recent_user else "",
                    "pred": memory.pending_pred,
                    "confirmed_state": state,
                    "intervention": action,
                }
            )
        memory.last_care_at = now
        return _action_slip(state, score, action, memory.recent_user, ask=False)

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

    def classify(self, text: str) -> EmotionScore:
        return self._predict(text)

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


def _decide_state(text: str, score: EmotionScore, fine_label: str | None = None) -> str:
    if score.pred == "sad" and fine_label in {"低落", "有压力", "委屈", "说不清"}:
        if fine_label == "有压力":
            return "stressed"
        if fine_label == "委屈":
            return "wronged"
        if fine_label == "低落":
            return "low"
        return "low" if score.confidence >= ASK_AT["low"] else "unclear"
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


ACTION_TASK = {
    "listen": "下一步已经定为：听。只用他的原话接住这些内容，让他继续说。不要给建议，不要讲故事，不要改成别的做法。",
    "unpack": "下一步已经定为：拆一小步。只帮他把原话里最压的一件事收成下一步能做的一小步，别的不展开。不要讲故事。",
    "stay": "下一步已经定为：陪着。轻轻接住原话，留在这儿。不要追问，不要建议，不要讲故事。",
    "reflect": "下一步已经定为：听完并复述。用他的原话复述他在气什么。不要劝他冷静，不要讲道理，不要讲故事，不要给建议。",
    "story": "下一步已经定为：讲一个短故事。故事要接上他刚才说的事，讲完停下。不要分析情绪，不要说教。",
}


def _explicit_action(state: str, text: str) -> str | None:
    if _wants_story(text) and state in {"low", "wronged"}:
        return "story"
    if _wants_listen(text):
        return "listen"
    if "理一下" in text:
        return "unpack"
    return None


COARSE_NAME = {
    "happy": "高兴",
    "angry": "愤怒",
    "sad": "悲伤",
    "fear": "恐惧",
    "surprise": "惊奇",
    "neutral": "无情绪",
}


DECIDE_LINE = "请你判断这一轮该做什么，并直接回复。做法要贴着原话，一轮只做一件事。不要改粗分类。"


def _runner_up(score: EmotionScore) -> tuple[str, float] | None:
    ranked = sorted(score.probs.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2 or ranked[1][1] < 0.20:
        return None
    return ranked[1]


def _choice_slip(state: str, score: EmotionScore, lines: list[str], ask: bool) -> Slip:
    quoted = "\n".join(f"- {line}" for line in lines[-4:])
    body = [_coarse_line(score)]
    if score.pred == "sad" and state in {"low", "stressed", "wronged"}:
        body.append(f"细判断：{STATE_NAME[state]}。MacBERT 没有这一类，这一层看原话补充。")
    second = _runner_up(score)
    if second is not None:
        name, prob = second
        body.append(f"第二名：{COARSE_NAME.get(name, name)}（{name}，概率 {prob:.2f}）。回复时可以参考，但不能改掉粗分类。")
    body.extend([f"要融入的原话：\n{quoted}", DECIDE_LINE])
    return Slip(state, True, _slip(body), ask=ask)


def _coarse_line(score: EmotionScore) -> str:
    return (
        f"粗分类：{COARSE_NAME.get(score.pred, score.pred)}"
        f"（{score.pred}，置信度 {score.confidence:.2f}）。不要改这个粗分类。"
    )


def _action_slip(state: str, score: EmotionScore, action: str, lines: list[str], ask: bool) -> Slip:
    quoted = "\n".join(f"- {line}" for line in lines[-4:])
    body = [
        (
            f"粗分类：{COARSE_NAME.get(score.pred, score.pred)}"
            f"（{score.pred}，置信度 {score.confidence:.2f}）。不要改这个粗分类。"
        ),
    ]
    if score.pred == "sad" and state in {"low", "stressed", "wronged"}:
        body.append(f"细判断：{STATE_NAME[state]}。MacBERT 没有这一类，这一层看原话补充。")
    body.extend(
        [
            f"要融入的原话：\n{quoted}",
            ACTION_TASK[action],
        ]
    )
    return Slip(state, True, _slip(body), ask=ask)


def _remember(memory: Memory, text: str) -> None:
    if not text:
        return
    memory.recent_user.append(text)
    del memory.recent_user[:-6]


def _wants_story(text: str) -> bool:
    return "故事" in text


def _wants_stop(text: str) -> bool:
    return any(word in text for word in ("不想说", "别说了", "不说了"))


def _wants_listen(text: str) -> bool:
    return any(word in text for word in ("听着", "你听", "听我说"))


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


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _contentless(text: str) -> bool:
    return not any("\u4e00" <= char <= "\u9fff" or char.isalnum() for char in text)
