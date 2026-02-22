"""
InsightRadar — API：输出为 JSON / Markdown，供研究者与产品使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ..core.schemas import NormalizedDocument, Topic, InsightOutput
from ..ingestion import load_rss_sources, scrape_web_sources, load_local_documents
from ..processing import get_embedder, build_topics, compute_trend_score, analyze_sentiment, aggregate_sentiment
from ..insight import build_timeline, generate_insight
from ..storage import get_db, init_db, save_documents, save_topics, save_insights
from ..storage.database import DEFAULT_DB_PATH


app = FastAPI(
    title="InsightRadar API",
    description="从信息洪水中提炼趋势结构，面向研究者/产品/投资/创作者的洞察。",
    version="0.1.0",
)


def _load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "config" / "sources.yaml"
    if not config_path.is_file():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@app.get("/")
def root():
    return {
        "name": "InsightRadar",
        "description": "从信息洪水中提炼趋势结构，而非摘要新闻。",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(
    rss: bool = True,
    web: bool = True,
    local: bool = False,
):
    """执行一次摄取：RSS / 网页 / 本地。返回新摄取文档数。"""
    config = _load_config()
    docs: list[NormalizedDocument] = []
    if rss and config.get("rss"):
        docs.extend(load_rss_sources(config["rss"]))
    if web and config.get("web_scrape"):
        docs.extend(scrape_web_sources(config["web_scrape"]))
    if local and config.get("local"):
        base = config["local"].get("base_path", "./data/local")
        exts = config["local"].get("extensions", [".txt", ".md"])
        rec = config["local"].get("recursive", True)
        docs.extend(load_local_documents(base, exts, rec))
    if not docs:
        return {"ingested": 0, "documents": []}
    session = get_db()
    save_documents(session, docs)
    session.close()
    return {"ingested": len(docs), "documents": [d.model_dump() for d in docs[:20]]}


@app.get("/topics")
def list_topics():
    """返回已存储的 Topic 列表（从 DB 读，若无可返回空）。"""
    session = get_db()
    try:
        from sqlalchemy import text
        rows = session.execute(text("SELECT topic_id, topic_title, summary, time_span FROM topics")).fetchall()
        return {"topics": [{"topic_id": r[0], "topic_title": r[1], "summary": r[2], "time_span": r[3]} for r in rows]}
    except Exception:
        return {"topics": []}
    finally:
        session.close()


@app.get("/insights")
def list_insights():
    """返回已存储的 Insight 列表。"""
    session = get_db()
    try:
        from sqlalchemy import text
        rows = session.execute(text("SELECT insight_id, topic_id, title, strength_explanation, risk_opportunity, generated_at FROM insights")).fetchall()
        return {"insights": [{"insight_id": r[0], "topic_id": r[1], "title": r[2], "strength_explanation": r[3], "risk_opportunity": r[4], "generated_at": r[5]} for r in rows]}
    except Exception:
        return {"insights": []}
    finally:
        session.close()


@app.get("/run-pipeline")
def run_pipeline(
    min_cluster_size: int = Query(3, ge=2),
    generate_insights: bool = True,
):
    """
    运行完整管道：从 DB 取文档 → embedding → 聚类 → 重命名 → TrendScore → 仅当显著变化时生成 Insight。
    返回 topics 与 insights 摘要。
    """
    session = get_db()
    try:
        from sqlalchemy import text
        rows = session.execute(text("SELECT doc_id, source, timestamp, author, raw_text, title, url, meta FROM documents")).fetchall()
    except Exception:
        return {"error": "Failed to load documents", "topics": [], "insights": []}
    finally:
        session.close()

    if not rows:
        return {"message": "No documents in DB. Run /ingest first.", "topics": [], "insights": []}

    import json
    from datetime import datetime
    docs = []
    for r in rows:
        meta = json.loads(r[7]) if isinstance(r[7], str) else (r[7] or {})
        ts = r[2]
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.utcnow()
        docs.append(NormalizedDocument(doc_id=r[0], source=r[1], timestamp=ts, author=r[3], raw_text=r[4], title=r[5], url=r[6], meta=meta))

    embedder = get_embedder()
    embeddings = embedder.embed_documents(docs)
    config = _load_config()
    proc = config.get("processing") or {}
    min_size = proc.get("cluster_min_size", min_cluster_size)
    topics = build_topics(docs, embeddings=embeddings, min_cluster_size=min_size, use_llm_rename=True)
    if not topics:
        return {"topics": [], "insights": [], "message": "No topics formed."}

    session = get_db()
    save_topics(session, topics)
    # 恢复 documents 到 topic 以便 scoring 和 insight
    for t in topics:
        t.documents = [d for d in docs if d.doc_id in t.doc_ids]

    trend_scores = [compute_trend_score(t) for t in topics]
    significant_delta = proc.get("trend_significant_delta", 15)
    insights: list[InsightOutput] = []
    for topic, score in zip(topics, trend_scores):
        if not generate_insights:
            continue
        timeline = build_timeline(topic)
        sent_results = [analyze_sentiment(d) for d in topic.documents]
        sent_agg = aggregate_sentiment(sent_results) if sent_results else None
        insight = generate_insight(topic, trend_score=score, timeline=timeline, sentiment_agg=sent_agg)
        insights.append(insight)
    save_insights(session, insights)
    session.close()

    return {
        "topics": [{"topic_id": t.topic_id, "topic_title": t.topic_title, "time_span": t.time_span} for t in topics],
        "insights": [{"insight_id": i.insight_id, "title": i.title, "trend_score": i.trend_score.score if i.trend_score else None} for i in insights],
    }


@app.get("/insight/{insight_id}", response_class=PlainTextResponse)
def get_insight_markdown(insight_id: str):
    """返回单个 Insight 的 Markdown 格式，便于阅读与导出。"""
    session = get_db()
    try:
        from sqlalchemy import text
        row = session.execute(text("SELECT insight_id, topic_id, title, strength_explanation, risk_opportunity, key_citations, sentiment_summary, next_steps, generated_at FROM insights WHERE insight_id = :id"), {"id": insight_id}).fetchone()
    except Exception:
        return f"Insight {insight_id} not found."
    finally:
        session.close()
    if not row:
        return f"Insight {insight_id} not found."
    import json
    citations = json.loads(row[5]) if isinstance(row[5], str) else (row[5] or [])
    next_steps = json.loads(row[7]) if isinstance(row[7], str) else (row[7] or [])
    md = f"""# {row[2]}

**Topic ID:** {row[1]}  
**Generated:** {row[8]}

## 趋势强度解释
{row[3]}

## 风险 / 机会提示
{row[4]}

## 情绪总结
{row[6] or '—'}

## 可能下一步
"""
    for s in next_steps:
        md += f"- {s}\n"
    md += "\n## 关键引用\n"
    for c in citations[:10]:
        md += f"- **{c.get('title', c.get('source', ''))}** — {c.get('snippet', '')[:200]}…\n"
    return md
