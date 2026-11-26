"""加载交接包里的 SMP2020 MacBERT 六分类头。"""

from pathlib import Path

from emotion.engine import EmotionScore


class MacBertClassifier:
    """延迟加载训练好的 MacBERT。聊天模型不参与这次判断。"""

    def __init__(self, base_dir: str, checkpoint: str, device: str = "cpu") -> None:
        self.base_dir = str(base_dir or "").strip()
        self.checkpoint = str(checkpoint or "").strip()
        self.device = device or "cpu"
        self._model = None
        if not self.base_dir or not self.checkpoint:
            raise FileNotFoundError("请填写 macbert_base_dir 和 macbert_checkpoint")
        if not Path(self.base_dir).is_dir():
            raise FileNotFoundError(f"找不到 MacBERT 基座目录：{self.base_dir}")
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"找不到 MacBERT 权重：{self.checkpoint}")

    def predict(self, text: str) -> EmotionScore:
        if self._model is None:
            self._model = _load(self.base_dir, self.checkpoint, self.device)
        return EmotionScore.from_payload(self._model.predict(text))


def _load(base_dir: str, checkpoint: str, device: str):
    try:
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("MacBERT 需要在 AstrBot 的 Python 里安装 torch 和 transformers") from exc

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = Path(base_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
            self.tokenizer.truncation_side = "right"
            config = AutoConfig.from_pretrained(base, local_files_only=True)
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
            self.head = nn.Linear(config.hidden_size, 6)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=True)
            self.to(device=torch.device(device), dtype=torch.float32).eval()
            self._torch = torch

        def predict(self, text: str) -> dict:
            tokens = self.tokenizer(
                text,
                max_length=192,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            batch = tokens.to(next(self.parameters()).device)
            with self._torch.inference_mode():
                hidden = self.encoder(**batch).last_hidden_state[:, 0]
                logits = self.head(hidden)[0].float().cpu()
            probabilities = logits.softmax(dim=-1).tolist()
            labels = ["happy", "angry", "sad", "fear", "surprise", "neutral"]
            top = max(range(6), key=probabilities.__getitem__)
            return {
                "pred": labels[top],
                "probs": dict(zip(labels, probabilities, strict=True)),
                "confidence": probabilities[top],
            }

    return Net()
