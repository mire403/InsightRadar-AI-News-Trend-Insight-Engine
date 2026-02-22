"""
InsightRadar — 趋势时间线：起点事件、关键转折、情绪拐点。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..core.schemas import NormalizedDocument, Topic, TimelineEvent, SentimentResult
from ..processing.sentiment import analyze_sentiment, aggregate_sentiment


def build_timeline(
    topic: Topic,
    max_events: int = 10,
) -> list[TimelineEvent]:
    """
    对每个 Topic 生成时间线：起点事件、关键转折节点、情绪变化拐点。
    """
    docs = topic.documents or []
    if not docs:
        return []

    sorted_docs = sorted(docs, key=lambda d: d.timestamp)
    events: list[TimelineEvent] = []

    # 起点事件：最早的一批文档
    start = sorted_docs[0]
    events.append(
        TimelineEvent(
            event_id=f"{topic.topic_id}_start",
            label="起点事件",
            description=start.title or start.raw_text[:300] or "（无标题）",
            timestamp=start.timestamp,
            date_str=start.timestamp.strftime("%Y-%m-%d"),
            doc_ids=[start.doc_id],
        )
    )

    # 按时间分段，找情绪拐点（前后半段情绪差异大）
    n = len(sorted_docs)
    if n >= 4:
        mid = n // 2
        early = sorted_docs[:mid]
        late = sorted_docs[mid:]
        r_early = aggregate_sentiment([analyze_sentiment(d) for d in early])
        r_late = aggregate_sentiment([analyze_sentiment(d) for d in late])
        strong = ["hype", "anger", "optimism", "concern"]
        e_strong = sum(r_early.dimensions.get(s, 0) or 0 for s in strong)
        l_strong = sum(r_late.dimensions.get(s, 0) or 0 for s in strong)
        if abs(l_strong - e_strong) > 0.2:
            turn = sorted_docs[mid]
            events.append(
                TimelineEvent(
                    event_id=f"{topic.topic_id}_sentiment_turn",
                    label="情绪拐点",
                    description=f"讨论情绪从 {'较强' if e_strong > l_strong else '较弱'} 转为 {'较强' if l_strong > e_strong else '较弱'}",
                    timestamp=turn.timestamp,
                    date_str=turn.timestamp.strftime("%Y-%m-%d"),
                    doc_ids=[turn.doc_id],
                    sentiment_snapshot=r_late,
                )
            )

    # 关键转折：取中间某点或讨论量突增点（这里简化为中点附近）
    if n >= 3:
        idx = n // 2
        turn_doc = sorted_docs[idx]
        events.append(
            TimelineEvent(
                event_id=f"{topic.topic_id}_turn_{idx}",
                label="关键转折",
                description=turn_doc.title or turn_doc.raw_text[:200] or "（无标题）",
                timestamp=turn_doc.timestamp,
                date_str=turn_doc.timestamp.strftime("%Y-%m-%d"),
                doc_ids=[turn_doc.doc_id],
            )
        )

    # 最新节点
    if n > 1:
        latest = sorted_docs[-1]
        events.append(
            TimelineEvent(
                event_id=f"{topic.topic_id}_latest",
                label="最新进展",
                description=latest.title or latest.raw_text[:300] or "（无标题）",
                timestamp=latest.timestamp,
                date_str=latest.timestamp.strftime("%Y-%m-%d"),
                doc_ids=[latest.doc_id],
            )
        )

    # 去重并限制数量，按时间排序
    seen = set()
    unique: list[TimelineEvent] = []
    for e in events:
        if e.event_id not in seen:
            seen.add(e.event_id)
            unique.append(e)
    unique.sort(key=lambda x: (x.timestamp or datetime.min))
    return unique[:max_events]
