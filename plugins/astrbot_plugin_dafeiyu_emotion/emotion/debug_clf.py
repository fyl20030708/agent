"""没有 MacBERT 权重时，用关键词走通纸条。不能当成情绪识别结果。"""

from emotion.engine import EmotionScore


class DebugClassifier:
    """本地联调用的假分类器。配置改回 macbert 后不再使用。"""

    def predict(self, text: str) -> EmotionScore:
        if any(word in text for word in ("压力", "焦虑", "睡不着", "身心俱疲", "忙不过来")):
            return EmotionScore.from_probs({"sad": 0.58, "angry": 0.22, "neutral": 0.20})
        if any(word in text for word in ("害怕", "好怕", "心跳")):
            return EmotionScore.from_probs({"fear": 0.88, "sad": 0.12})
        if any(word in text for word in ("过分", "生气", "火大")):
            return EmotionScore.from_probs({"angry": 0.90, "neutral": 0.10})
        if any(word in text for word in ("不想", "难过", "低落", "答辩")):
            return EmotionScore.from_probs({"sad": 0.86, "neutral": 0.14})
        if any(word in text for word in ("开心", "高兴", "阳光")):
            return EmotionScore.from_probs({"happy": 0.91, "neutral": 0.09})
        if "电影" in text or "剧情" in text:
            return EmotionScore.from_probs({"sad": 0.90, "neutral": 0.10})
        return EmotionScore.from_probs({"neutral": 0.80, "happy": 0.20})
