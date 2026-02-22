"""
InsightRadar — 所有 LLM 调用集中封装。
便于切换模型、统一 prompt、限流与缓存。
"""

from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """统一 LLM 接口：OpenAI 兼容 API。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """单轮文本补全，返回模型输出字符串。"""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package required. pip install openai")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """要求模型返回合法 JSON，解析为 dict。"""
        import json

        sys = (system or "") + "\n\nYou must respond with a single valid JSON object, no markdown."
        raw = self.complete(prompt, system=sys, max_tokens=max_tokens, **kwargs)
        # 允许被 ```json ... ``` 包裹
        if "```" in raw:
            start = raw.find("```")
            if "json" in raw[: start + 10]:
                start = raw.find("\n", start) + 1
            end = raw.find("```", start)
            raw = raw[start:end] if end > start else raw[start:]
        return json.loads(raw)


# 单例，便于全局复用与缓存装饰器使用
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def set_llm_client(client: LLMClient) -> None:
    global _default_client
    _default_client = client
