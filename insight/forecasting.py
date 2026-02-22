"""
InsightRadar — 可能的下一步发展：概率性推断，非预测。
不预测具体金融价格，不提供投资建议。
"""

from __future__ import annotations

from typing import Optional

from ..core.schemas import Topic, TrendScore, SentimentResult
from ..core.llm_client import get_llm_client


def suggest_next_steps(
    topic: Topic,
    trend_score: Optional[TrendScore] = None,
    sentiment_agg: Optional[SentimentResult] = None,
    max_steps: int = 5,
) -> list[str]:
    """
    基于主题、趋势强度与情绪，生成「可能的下一步发展」列表。
    表述为概率性、情景性描述，明确非预测、非投资建议。
    """
    summary = topic.summary or topic.topic_title
    docs_snippet = ""
    for d in (topic.documents or [])[:5]:
        docs_snippet += (d.title or "") + " " + (d.raw_text or "")[:200] + "\n"
    score_text = f"趋势强度约 {trend_score.score:.0f}/100。" if trend_score else ""
    sent_text = ""
    if sentiment_agg and sentiment_agg.label:
        sent_text = f"讨论情绪概括：{sentiment_agg.label}"

    prompt = f"""基于以下主题信息，列出 3–5 条「可能的下一步发展」。
要求：表述为概率性、情景性（如「可能…」「若…则…」），不要预测具体价格或给出投资建议。
仅作研究/产品/创作参考。

主题：{topic.topic_title}
摘要：{summary}
{score_text}
{sent_text}

文档片段：
{docs_snippet[:2000]}

请用 JSON 返回：{{"next_steps": ["第一条", "第二条", ...]}}"""

    try:
        llm = get_llm_client()
        out = llm.complete_json(prompt, max_tokens=512)
        steps = out.get("next_steps") or []
        if isinstance(steps, list):
            return [str(s).strip() for s in steps[:max_steps] if s and str(s).strip()]
    except Exception:
        pass
    return [
        "讨论可能随新政策或事件继续演化，建议持续跟踪多源信息。",
        "本系统不预测具体结果，仅提示可能方向供研究者与创作者参考。",
    ]
