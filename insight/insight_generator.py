"""
InsightRadar — 洞察级输出生成：一句标题、强度解释、风险/机会、关键引用。
不是摘要，是可回溯的洞察。
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

from ..core.schemas import (
    Topic,
    TrendScore,
    InsightOutput,
    TimelineEvent,
    SentimentResult,
)
from ..processing.sentiment import aggregate_sentiment, analyze_sentiment, sentiment_summary_label
from .timeline_builder import build_timeline
from .forecasting import suggest_next_steps


def _insight_id(topic_id: str, at: datetime) -> str:
    raw = f"{topic_id}:{at.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _key_citations(topic: Topic, max_citations: int = 5) -> list[dict]:
    """从 Topic 中抽取关键原始引用，可回溯。"""
    docs = topic.documents or []
    out = []
    for d in docs[:max_citations]:
        snippet = (d.raw_text or "")[:400].strip()
        out.append({
            "doc_id": d.doc_id,
            "source": d.source,
            "title": d.title,
            "snippet": snippet,
            "url": d.url,
            "timestamp": d.timestamp.isoformat() if d.timestamp else None,
        })
    return out


def generate_insight(
    topic: Topic,
    trend_score: Optional[TrendScore] = None,
    timeline: Optional[list[TimelineEvent]] = None,
    sentiment_agg: Optional[SentimentResult] = None,
    next_steps: Optional[list[str]] = None,
) -> InsightOutput:
    """
    对单个 Topic 生成最终 InsightOutput：
    一句洞察标题、趋势强度解释、风险/机会、关键引用、时间线、情绪总结、可能下一步。
    """
    if timeline is None:
        timeline = build_timeline(topic)
    if sentiment_agg is None and topic.documents:
        results = [analyze_sentiment(d) for d in topic.documents]
        sentiment_agg = aggregate_sentiment(results)
    sentiment_summary = sentiment_summary_label(sentiment_agg) if sentiment_agg else None
    if next_steps is None:
        next_steps = suggest_next_steps(topic, trend_score, sentiment_agg)

    # 标题与强度解释：若已有 trend_score 则用其 explanation，否则简短概括
    strength_explanation = (
        trend_score.explanation if trend_score and trend_score.explanation
        else f"主题「{topic.topic_title}」共 {len(topic.documents or [])} 篇文档，时间跨度 {topic.time_span}。"
    )
    title = topic.topic_title  # 洞察标题可直接用主题标题，或交给 LLM 再凝练

    risk_opportunity = _risk_opportunity_text(topic, trend_score, sentiment_agg)

    return InsightOutput(
        insight_id=_insight_id(topic.topic_id, datetime.utcnow()),
        topic_id=topic.topic_id,
        title=title,
        strength_explanation=strength_explanation,
        risk_opportunity=risk_opportunity,
        key_citations=_key_citations(topic),
        trend_score=trend_score,
        timeline=timeline,
        sentiment_summary=sentiment_summary,
        next_steps=next_steps,
        generated_at=datetime.utcnow(),
    )


def _risk_opportunity_text(
    topic: Topic,
    trend_score: Optional[TrendScore],
    sentiment_agg: Optional[SentimentResult],
) -> str:
    """生成风险/机会提示（概率性描述，非投资建议）。"""
    parts = []
    if trend_score:
        if trend_score.score >= 70:
            parts.append("趋势强度较高，关注度与扩散度上升，适合持续跟踪。")
        elif trend_score.score <= 30:
            parts.append("当前趋势强度较低，可能处于早期或衰减阶段。")
    if sentiment_agg and (sentiment_agg.polarization or 0) > 0.5:
        parts.append("讨论存在明显立场分裂，解读时需注意多角度。")
    if sentiment_agg:
        hype = (sentiment_agg.dimensions.get("hype") or 0)
        if hype > 0.5:
            parts.append("存在炒作情绪，需区分事实与情绪驱动。")
    if not parts:
        parts.append("建议结合多源与时间线自行判断，本系统不提供投资或决策建议。")
    return " ".join(parts)
