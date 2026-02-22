"""
InsightRadar — 结构化存储：文档、Topic、Insight 等中间结果可缓存、可复跑。
使用 SQLite + 简单表结构，便于本地运行。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from pydantic import BaseModel

from ..core.schemas import NormalizedDocument, Topic, TrendScore, InsightOutput

# 默认 DB 路径
DEFAULT_DB_PATH = "data/insight_radar.db"


def _json_serial(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(type(obj))


def get_engine(db_path: Optional[str] = None):
    path = db_path or DEFAULT_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


def init_db(db_path: Optional[str] = None) -> None:
    """创建表结构。"""
    engine = get_engine(db_path)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT,
                timestamp TEXT,
                author TEXT,
                raw_text TEXT,
                title TEXT,
                url TEXT,
                meta TEXT,
                created_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS topics (
                topic_id TEXT PRIMARY KEY,
                topic_title TEXT,
                summary TEXT,
                doc_ids TEXT,
                time_span TEXT,
                centroid_id TEXT,
                meta TEXT,
                created_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trend_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id TEXT,
                score REAL,
                volume_change REAL,
                acceleration REAL,
                sentiment_shift REAL,
                source_diffusion REAL,
                explanation TEXT,
                computed_at TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS insights (
                insight_id TEXT PRIMARY KEY,
                topic_id TEXT,
                title TEXT,
                strength_explanation TEXT,
                risk_opportunity TEXT,
                key_citations TEXT,
                trend_score TEXT,
                timeline TEXT,
                sentiment_summary TEXT,
                next_steps TEXT,
                generated_at TEXT
            )
        """))
        conn.commit()


def get_db(db_path: Optional[str] = None) -> Session:
    engine = get_engine(db_path)
    init_db(db_path)
    return sessionmaker(bind=engine, autocommit=False, autobegin=True)()


def save_documents(session: Session, documents: list[NormalizedDocument]) -> None:
    now = datetime.utcnow().isoformat()
    for d in documents:
        session.execute(
            text("""
                INSERT OR REPLACE INTO documents (doc_id, source, timestamp, author, raw_text, title, url, meta, created_at)
                VALUES (:doc_id, :source, :timestamp, :author, :raw_text, :title, :url, :meta, :created_at)
            """),
            {
                "doc_id": d.doc_id,
                "source": d.source,
                "timestamp": d.timestamp.isoformat() if d.timestamp else None,
                "author": d.author,
                "raw_text": d.raw_text,
                "title": d.title,
                "url": d.url,
                "meta": json.dumps(d.meta, default=_json_serial),
                "created_at": now,
            },
        )
    session.commit()


def save_topics(session: Session, topics: list[Topic]) -> None:
    now = datetime.utcnow().isoformat()
    for t in topics:
        session.execute(
            text("""
                INSERT OR REPLACE INTO topics (topic_id, topic_title, summary, doc_ids, time_span, centroid_id, meta, created_at)
                VALUES (:topic_id, :topic_title, :summary, :doc_ids, :time_span, :centroid_id, :meta, :created_at)
            """),
            {
                "topic_id": t.topic_id,
                "topic_title": t.topic_title,
                "summary": t.summary,
                "doc_ids": json.dumps(t.doc_ids),
                "time_span": json.dumps(t.time_span),
                "centroid_id": t.centroid_id,
                "meta": json.dumps(t.meta, default=_json_serial),
                "created_at": now,
            },
        )
    session.commit()


def save_insights(session: Session, insights: list[InsightOutput]) -> None:
    for i in insights:
        session.execute(
            text("""
                INSERT OR REPLACE INTO insights (
                    insight_id, topic_id, title, strength_explanation, risk_opportunity,
                    key_citations, trend_score, timeline, sentiment_summary, next_steps, generated_at
                ) VALUES (
                    :insight_id, :topic_id, :title, :strength_explanation, :risk_opportunity,
                    :key_citations, :trend_score, :timeline, :sentiment_summary, :next_steps, :generated_at
                )
            """),
            {
                "insight_id": i.insight_id,
                "topic_id": i.topic_id,
                "title": i.title,
                "strength_explanation": i.strength_explanation,
                "risk_opportunity": i.risk_opportunity,
                "key_citations": json.dumps(i.key_citations, default=_json_serial),
                "trend_score": i.trend_score.model_dump_json() if i.trend_score else None,
                "timeline": json.dumps([e.model_dump() for e in i.timeline], default=_json_serial),
                "sentiment_summary": i.sentiment_summary,
                "next_steps": json.dumps(i.next_steps),
                "generated_at": i.generated_at.isoformat() if i.generated_at else None,
            },
        )
    session.commit()
