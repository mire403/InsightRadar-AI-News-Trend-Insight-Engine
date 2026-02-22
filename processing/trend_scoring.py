"""
InsightRadar — 趋势强度建模：讨论量变化、加速度、情绪变化、来源扩散度 → TrendScore 0–100。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from ..core.schemas import NormalizedDocument, Topic, TrendScore, SentimentResult
from ..processing.sentiment import analyze_sentiment, aggregate_sentiment


def _volume_series(docs: list[NormalizedDocument], bin_days: int = 1) -> list[float]:
    """按时间桶统计文档数，得到讨论量时间序列。"""
    if not docs:
        return []
    base = min(d.timestamp for d in docs)
    buckets: dict[int, int] = defaultdict(int)
    for d in docs:
        delta = (d.timestamp - base).total_seconds() / (86400 * max(1, bin_days))
        buckets[int(delta)] += 1
    if not buckets:
        return [0.0]
    max_bucket = max(buckets.keys())
    return [float(buckets.get(i, 0)) for i in range(max_bucket + 1)]


def _volume_change(series: list[float]) -> float:
    """讨论量变化：后半段均值 / 前半段均值，或线性斜率归一化。"""
    if len(series) < 2:
        return 1.0
    mid = len(series) // 2
    first = np.mean(series[: mid + 1]) or 1e-6
    second = np.mean(series[mid:]) or 1e-6
    return second / first


def _acceleration(series: list[float]) -> float:
    """加速度：二阶差分或后半段斜率 vs 前半段斜率。"""
    if len(series) < 3:
        return 0.0
    arr = np.array(series, dtype=float)
    mid = len(arr) // 2
    first_half = arr[: mid + 1]
    second_half = arr[mid:]
    s1 = np.polyfit(np.arange(len(first_half)), first_half, 1)[0] if len(first_half) > 1 else 0
    s2 = np.polyfit(np.arange(len(second_half)), second_half, 1)[0] if len(second_half) > 1 else 0
    return float(s2 - s1)


def _sentiment_shift(docs: list[NormalizedDocument]) -> float:
    """情绪倾向变化：从中性 → 强烈。按时间排序后比较前后半段情绪强度。"""
    if len(docs) < 2:
        return 0.0
    sorted_docs = sorted(docs, key=lambda d: d.timestamp)
    mid = len(sorted_docs) // 2
    early = sorted_docs[:mid]
    late = sorted_docs[mid:]
    r_early = aggregate_sentiment([analyze_sentiment(d) for d in early])
    r_late = aggregate_sentiment([analyze_sentiment(d) for d in late])
    strong = ["hype", "anger", "optimism", "concern"]
    e_strong = sum(r_early.dimensions.get(s, 0) or 0 for s in strong)
    l_strong = sum(r_late.dimensions.get(s, 0) or 0 for s in strong)
    return min(1.0, max(0.0, l_strong - e_strong))


def _source_diffusion(docs: list[NormalizedDocument]) -> float:
    """来源扩散度 0–1：是否从小圈层扩散（来源种类数 / 文档数 比例，归一化）。"""
    if not docs:
        return 0.0
    sources = len(set(d.source for d in docs))
    n = len(docs)
    # 来源越多、文档越多，扩散度越高；简单用 sources / sqrt(n) 再 cap 到 1
    raw = sources / (n ** 0.5) if n else 0
    return min(1.0, raw / 2.0)


def compute_trend_score(
    topic: Topic,
    volume_change: Optional[float] = None,
    acceleration: Optional[float] = None,
    sentiment_shift: Optional[float] = None,
    source_diffusion: Optional[float] = None,
) -> TrendScore:
    """
    为单个 Topic 计算 TrendScore (0–100)。
    若未传入子指标则根据 topic.documents 计算。
    """
    docs = topic.documents or []
    series = _volume_series(docs)

    if volume_change is None:
        volume_change = _volume_change(series)
    if acceleration is None:
        acceleration = _acceleration(series)
    if sentiment_shift is None:
        sentiment_shift = _sentiment_shift(docs)
    if source_diffusion is None:
        source_diffusion = _source_diffusion(docs)

    # 归一化到 0–1 再映射到 0–100
    v_norm = min(1.0, max(0.0, (volume_change - 0.5) / 1.5 + 0.5))  # 1.0 -> 0.5, 2.0 -> 1
    a_norm = min(1.0, max(0.0, (acceleration + 2) / 4))  # 简单平移缩放
    s_norm = min(1.0, max(0.0, sentiment_shift))
    d_norm = min(1.0, max(0.0, source_diffusion))

    score = (v_norm * 0.3 + a_norm * 0.25 + s_norm * 0.25 + d_norm * 0.2) * 100
    score = round(min(100.0, max(0.0, score)), 1)

    explanation = (
        f"讨论量变化={volume_change:.2f}, 加速度={acceleration:.2f}, "
        f"情绪变化={sentiment_shift:.2f}, 来源扩散={source_diffusion:.2f}"
    )

    return TrendScore(
        topic_id=topic.topic_id,
        score=score,
        volume_change=volume_change,
        acceleration=acceleration,
        sentiment_shift=sentiment_shift,
        source_diffusion=source_diffusion,
        explanation=explanation,
        computed_at=datetime.utcnow(),
    )
