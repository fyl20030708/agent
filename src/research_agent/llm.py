"""LLM client settings from environment (OpenAI-compatible / DeepSeek)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ClientSettings:
    api_key: str
    base_url: str
    model: str


def load_settings() -> ClientSettings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env (DeepSeek key is fine)")
    return ClientSettings(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
    )
