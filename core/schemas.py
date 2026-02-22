"""
InsightRadar — 统一数据模型。
所有 ingestion 输出、processing / insight 中间与最终结果均基于此。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ----- Ingestion: 统一文档 -----


class NormalizedDocument(BaseModel):
    """所有输入源统一转换为此格式。"""

    doc_id: str = Field(..., description="唯一 ID，可由 source + timestamp 生成")
    source: str = Field(..., description="来源标识，如 rss:tech_crunch, local:meeting.md")
    timestamp: datetime = Field(..., description="文档时间")
    author: Optional[str] = Field(None, description="作者或来源账号")
    raw_text: str = Field(..., description="原始正文")
    title: Optional[str] = Field(None, description="标题（若有）")
    url: Optional[str] = Field(None, description="原文链接")
    meta: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ----- 情绪维度（非简单正负面） -----


class SentimentDimension(str, Enum):
    NEUTRAL = "neutral"
    CONCERN = "concern"
    OPTIMISM = "optimism"
    ANGER = "anger"
    HYPE = "hype"


class SentimentResult(BaseModel):
    """单文档或聚合情绪结果。"""

    dimensions: dict[str, float] = Field(
        ...,
        description="各维度强度 0–1，如 neutral: 0.3, hype: 0.6",
    )
    polarization: Optional[float] = Field(
        None,
        description="立场分裂程度 0–1，越高越极化",
    )
    label: Optional[str] = Field(
        None,
        description="主情绪标签，如 '被情绪推着走的趋势'",
    )


# ----- Topic（语义主题） -----


class Topic(BaseModel):
    """聚类 + LLM 重命名后的主题事件。"""

    topic_id: str = Field(..., description="主题唯一 ID")
    topic_title: str = Field(..., description="人能读懂的标题，如 'AI 芯片出口管制升级'")
    summary: str = Field(..., description="主题摘要")
    documents: list[NormalizedDocument] = Field(default_factory=list)
    time_span: list[str] = Field(
        ...,
        description="时间范围 [start_date, end_date]，如 ['2026-01-10', '2026-02-01']",
    )
    doc_ids: list[str] = Field(default_factory=list, description="关联文档 ID，便于去重")
    centroid_id: Optional[str] = Field(None, description="聚类中心文档 ID")
    meta: dict[str, Any] = Field(default_factory=dict)


# ----- 趋势强度 -----


class TrendScore(BaseModel):
    """每个 Topic 的趋势指标，0–100。"""

    topic_id: str
    score: float = Field(..., ge=0, le=100)
    volume_change: Optional[float] = Field(
        None,
        description="讨论量变化（时间序列斜率或比值）",
    )
    acceleration: Optional[float] = Field(
        None,
        description="加速度（增长是否异常）",
    )
    sentiment_shift: Optional[float] = Field(
        None,
        description="情绪倾向变化强度（从中性→强烈）",
    )
    source_diffusion: Optional[float] = Field(
        None,
        description="来源扩散度 0–1，是否从小圈层扩散",
    )
    explanation: Optional[str] = Field(None, description="强度解释")
    computed_at: datetime = Field(default_factory=datetime.utcnow)


# ----- 时间线 -----


class TimelineEvent(BaseModel):
    """时间线上的节点。"""

    event_id: str
    label: str = Field(..., description="如 '起点事件' / '关键转折' / '情绪拐点'")
    description: str
    timestamp: Optional[datetime] = None
    date_str: Optional[str] = None
    doc_ids: list[str] = Field(default_factory=list)
    sentiment_snapshot: Optional[SentimentResult] = None


# ----- 洞察输出（最终形态） -----


class InsightOutput(BaseModel):
    """每个趋势的洞察级输出，不是摘要。"""

    insight_id: str
    topic_id: str
    title: str = Field(..., description="一句洞察标题")
    strength_explanation: str = Field(..., description="趋势强度解释：为什么在升/降")
    risk_opportunity: str = Field(..., description="风险 / 机会提示")
    key_citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="关键原始引用，可回溯：doc_id, source, snippet, url",
    )
    trend_score: Optional[TrendScore] = None
    timeline: list[TimelineEvent] = Field(default_factory=list)
    sentiment_summary: Optional[str] = Field(
        None,
        description="如 '被冷静讨论的趋势' vs '被情绪推着走的趋势'",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="可能的下一步发展（概率性推断，非预测）",
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
