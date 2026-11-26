"""情绪版回复纸条：MacBERT 判断，选中的模型只负责说。"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "plugins" / "astrbot_plugin_dafeiyu_emotion"
sys.path.insert(0, str(ROOT))

from emotion.engine import EmotionScore, Engine
from emotion.inject import attach_instruction

T0 = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)


class Scripted:
    def __init__(self, score: EmotionScore) -> None:
        self.score = score
        self.calls = 0

    def predict(self, text: str) -> EmotionScore:
        self.calls += 1
        return self.score


def engine(score: EmotionScore) -> tuple[Engine, Scripted]:
    clf = Scripted(score)
    return Engine(clf), clf


def score(**probs: float) -> EmotionScore:
    return EmotionScore.from_probs(probs)


def test_happy_slip_keeps_the_chat_and_does_not_ask() -> None:
    bot, _ = engine(score(happy=0.91, neutral=0.09))
    slip = bot.handle("s", "今天阳光很好", T0)
    assert slip.action_state == "pleasant"
    assert slip.call_model is True
    assert slip.ask is False
    assert "不要问" in slip.instruction
    assert "不要改判" in slip.instruction


def test_sad_slip_decides_to_listen_before_the_model_speaks() -> None:
    bot, _ = engine(score(sad=0.86, neutral=0.14))
    slip = bot.handle("s", "答辩完了，什么都不想做", T0)
    assert slip.action_state == "low"
    assert slip.ask is True
    assert slip.call_model is True
    assert "下一步已经定为：听" in slip.instruction
    assert "什么都不想做" in slip.instruction
    assert "不要讲故事" in slip.instruction
    assert "不要改判" in slip.instruction
    assert "多想了" not in slip.instruction


def test_low_confidence_sadness_does_not_ask() -> None:
    bot, _ = engine(score(sad=0.62, neutral=0.38))
    slip = bot.handle("s", "有一点闷", T0)
    assert slip.action_state == "low"
    assert slip.ask is False
    assert "不要问" in slip.instruction


def test_stress_slip_does_not_offer_a_story() -> None:
    bot, _ = engine(score(sad=0.58, angry=0.22, neutral=0.20))
    slip = bot.handle("s", "最近压力好大，睡不着", T0)
    assert slip.action_state == "stressed"
    assert slip.ask is True
    assert "下一步已经定为：拆一小步" in slip.instruction
    assert "一件" in slip.instruction
    assert "不要讲故事" in slip.instruction


def test_exam_exhaustion_is_stress() -> None:
    bot, _ = engine(score(angry=0.34, sad=0.48, neutral=0.18))
    slip = bot.handle("s", "考研怎么办呀，已经被折磨得身心俱疲", T0)
    assert slip.action_state == "stressed"
    assert slip.ask is True


def test_fear_without_fear_words_does_not_ask() -> None:
    bot, _ = engine(score(fear=0.9, neutral=0.1))
    slip = bot.handle("s", "外面风好大", T0)
    assert slip.ask is False


def test_explicit_fear_asks_without_a_story() -> None:
    bot, _ = engine(score(fear=0.88, sad=0.12))
    slip = bot.handle("s", "我好害怕，心跳得厉害", T0)
    assert slip.action_state == "afraid"
    assert "下一步已经定为：陪着" in slip.instruction
    assert "心跳" in slip.instruction
    assert "不要讲故事" in slip.instruction


def test_movie_talk_is_not_the_owners_mood() -> None:
    bot, clf = engine(score(sad=0.9, neutral=0.1))
    slip = bot.handle("s", "这电影的结局好难过", T0)
    assert slip.action_state == "unclear"
    assert slip.ask is False
    assert clf.calls == 1


def test_follow_up_changes_the_next_action_and_keeps_earlier_words() -> None:
    bot, clf = engine(score(sad=0.9))
    first = bot.handle("s", "今天什么都不想做", T0)
    assert "下一步已经定为：听" in first.instruction
    follow = bot.handle("s", "也不知道明天该怎么跟导师讲", T0)
    assert "拆一小步" in follow.instruction
    assert "导师" in follow.instruction
    assert "什么都不想做" in follow.instruction
    assert clf.calls == 2
    told = bot.handle("s", "讲个短故事", T0)
    assert told.call_model is True
    assert "短故事" in told.instruction
    assert "不要分析情绪" in told.instruction
    assert bot.memory("s").episodes[-1]["intervention"] == "story"
    assert bot.memory("s").episodes[-1]["confirmed_state"] == "low"


def test_decline_is_a_negative_label_and_returns_to_chat() -> None:
    bot, _ = engine(score(sad=0.9, neutral=0.1))
    bot.handle("s", "今天什么都不想做", T0)
    slipped = bot.handle("s", "没有，我在讲电影", T0)
    assert slipped.ask is False
    assert "普通聊天" in slipped.instruction
    episode = bot.memory("s").episodes[-1]
    assert episode["pred"] == "sad"
    assert episode["confirmed_state"] == "none"
    assert episode["intervention"] == "ask"


def test_anger_is_reflected_in_the_owners_words() -> None:
    bot, _ = engine(score(angry=0.9, neutral=0.1))
    first = bot.handle("s", "他们太过分了", T0)
    assert "下一步已经定为：听完并复述" in first.instruction
    assert "太过分了" in first.instruction
    assert "不要讲故事" in first.instruction
    heard = bot.handle("s", "你听着就行", T0)
    assert "下一步已经定为：听" in heard.instruction
    assert "不要讲故事" in heard.instruction
    assert "不要给建议" in heard.instruction


def test_stress_confirmation_asks_the_model_to_unpack_one_thing() -> None:
    bot, _ = engine(score(sad=0.7, angry=0.2))
    first = bot.handle("s", "压力好大，忙不过来", T0)
    assert "拆一小步" in first.instruction
    clarified = bot.handle("s", "帮我理一下", T0)
    assert "一件" in clarified.instruction
    assert "不要讲故事" in clarified.instruction
    assert bot.memory("s").episodes[-1]["confirmed_state"] == "stressed"


def test_crisis_is_not_given_to_the_model() -> None:
    bot, clf = engine(score(sad=0.95))
    slip = bot.handle("s", "我不想活了", T0)
    assert slip.call_model is False
    assert slip.direct_reply is not None
    assert "010-82951332" in slip.direct_reply
    assert "故事" not in slip.direct_reply
    assert "情绪" not in slip.direct_reply
    assert clf.calls == 0


def test_care_waits_for_time_or_several_ordinary_turns() -> None:
    bot, _ = engine(score(sad=0.9))
    assert bot.handle("s", "今天什么都不想做", T0).ask is True
    bot.handle("s", "没有", T0)
    blocked = bot.handle("s", "还是不想动", T0 + timedelta(minutes=20))
    assert blocked.ask is False
    later = bot.handle("s", "还是不想动", T0 + timedelta(hours=3))
    assert later.ask is True


def test_many_ordinary_turns_allow_another_question() -> None:
    bot, _ = engine(score(sad=0.9))
    bot.handle("s", "今天什么都不想做", T0)
    bot.handle("s", "没有", T0)
    calm = score(neutral=0.9, happy=0.1)
    bot.classifier = Scripted(calm)
    for i in range(6):
        bot.handle("s", f"我们聊第{i}件普通事", T0)
    bot.classifier = Scripted(score(sad=0.9))
    assert bot.handle("s", "还是不想动", T0 + timedelta(minutes=30)).ask is True


def test_opt_out_stops_questions() -> None:
    bot, _ = engine(score(sad=0.95))
    bot.handle("s", "别问情绪", T0)
    slip = bot.handle("s", "今天什么都不想做", T0)
    assert bot.memory("s").ask_emotion is False
    assert slip.ask is False


def test_unrelated_reply_cancels_the_pending_question() -> None:
    bot, _ = engine(score(sad=0.9))
    bot.handle("s", "今天什么都不想做", T0)
    bot.classifier = Scripted(score(neutral=0.8, happy=0.2))
    slip = bot.handle("s", "对了晚饭吃面", T0)
    assert bot.memory("s").pending is None
    assert slip.ask is False
    assert slip.action_state == "calm"


def test_sessions_can_be_reloaded(tmp_path) -> None:
    path = tmp_path / "state.json"
    bot, _ = engine(score(sad=0.9))
    bot.store_path = path
    bot.handle("s", "今天什么都不想做", T0)
    bot.handle("s", "没有", T0)
    bot.save()
    restored, _ = engine(score(neutral=1))
    restored.store_path = path
    restored.load()
    assert restored.memory("s").episodes[-1]["confirmed_state"] == "none"


def test_normal_mode_keyword_asks_before_switching() -> None:
    bot, clf = engine(score(sad=0.9))
    slip = bot.suggest_switch("s", "最近压力好大", T0)
    assert bot.mode("s") == "normal"
    assert slip is not None
    assert slip.call_model is True
    assert "情绪版" in slip.instruction
    assert "不用" in slip.instruction
    assert clf.calls == 0


def test_chat_model_names_pressure_inside_macbert_sadness() -> None:
    bot, clf = engine(score(sad=0.86, neutral=0.14))
    slip = bot.handle("s", "答辩完了，什么都不想做", T0, fine_label="有压力")
    assert slip.action_state == "stressed"
    assert "粗分类：悲伤" in slip.instruction
    assert "细判断：有压力" in slip.instruction
    assert "拆一小步" in slip.instruction
    assert clf.calls == 1


def test_wronged_stays_a_fine_label_and_listens() -> None:
    bot, _ = engine(score(sad=0.8, neutral=0.2))
    slip = bot.handle("s", "评委当众否定我", T0, fine_label="委屈")
    assert slip.action_state == "wronged"
    assert "细判断：委屈" in slip.instruction
    assert "下一步已经定为：听" in slip.instruction


def test_fine_label_cannot_override_macbert_anger() -> None:
    bot, _ = engine(score(angry=0.9, neutral=0.1))
    slip = bot.handle("s", "他们太过分了", T0, fine_label="低落")
    assert slip.action_state == "angry"
    assert "细判断" not in slip.instruction


def test_parse_fine_label_keeps_four_words() -> None:
    from emotion.refine import parse_fine_label

    assert parse_fine_label("这句是有压力") == "有压力"
    assert parse_fine_label("低落") == "低落"
    assert parse_fine_label("高兴") == "说不清"


def test_happy_word_also_asks_about_switching() -> None:
    bot, _ = engine(score(sad=0.9))
    slip = bot.suggest_switch("s", "今天好开心", T0)
    assert slip is not None
    assert "情绪版" in slip.instruction
    assert bot.mode("s") == "normal"


def test_normal_chat_without_keywords_stays_quiet() -> None:
    bot, _ = engine(score(sad=0.9))
    assert bot.suggest_switch("s", "今天天气不错", T0) is None


def test_declining_the_switch_stays_in_normal_mode() -> None:
    bot, _ = engine(score(sad=0.9))
    bot.suggest_switch("s", "我有点难过", T0)
    stayed = bot.suggest_switch("s", "不用", T0)
    assert bot.mode("s") == "normal"
    assert stayed is not None
    assert "普通" in stayed.instruction
    again = bot.suggest_switch("s", "还是好焦虑", T0 + timedelta(hours=1))
    assert again is None


def test_switch_invite_waits_before_asking_again() -> None:
    bot, _ = engine(score(sad=0.9))
    assert bot.suggest_switch("s", "压力好大", T0) is not None
    bot.suggest_switch("s", "先吃饭吧", T0)
    assert bot.suggest_switch("s", "还是睡不着", T0 + timedelta(minutes=30)) is None
    later = bot.suggest_switch("s", "还是睡不着", T0 + timedelta(hours=3))
    assert later is not None
    assert "情绪版" in later.instruction


def test_emotion_mode_does_not_ask_to_switch() -> None:
    bot, _ = engine(score(sad=0.9))
    bot.set_mode("s", "emotion")
    assert bot.suggest_switch("s", "压力好大", T0) is None


def test_normal_mode_crisis_still_skips_the_model() -> None:
    bot, clf = engine(score(sad=0.9))
    slip = bot.suggest_switch("s", "我不想活了", T0)
    assert slip is not None
    assert slip.call_model is False
    assert slip.direct_reply is not None
    assert "010-82951332" in slip.direct_reply
    assert clf.calls == 0


def test_instruction_is_attached_beside_the_user_text() -> None:
    class Req:
        def __init__(self) -> None:
            self.prompt = "答辩完了"
            self.system_prompt = "你是大肥鱼"
            self.extra_user_content_parts = []

    req = Req()
    attach_instruction(req, "【本轮回复纸条】\n不要改判。")
    assert req.system_prompt == "你是大肥鱼"
    assert req.extra_user_content_parts
    assert "不要改判" in req.extra_user_content_parts[-1]["text"]
    assert req.extra_user_content_parts[-1]["temporary"] is True
