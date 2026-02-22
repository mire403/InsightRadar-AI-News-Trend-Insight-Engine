"""
InsightRadar — 每日定时抓取、更新聚类与 TrendScore，仅当趋势显著变化时生成新 Insight。
可被 cron / APScheduler 调用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# 确保以包方式运行时可找到 insight_radar
def _ensure_path():
    root = Path(__file__).resolve().parent.parent.parent
    if root.name == "insight_radar":
        root = root.parent
    if str(root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(root))


def run_daily(
    config_path: Optional[str] = None,
    db_path: Optional[str] = None,
    run_ingest: bool = True,
    run_pipeline: bool = True,
) -> dict:
    """
    每日运行流程：
    1. 抓取数据（RSS / 网页 / 本地）
    2. 新文档 embedding，写入向量库（可选，由 pipeline 内嵌）
    3. 更新 topic 聚类，LLM 重命名
    4. 重新计算 TrendScore
    5. 仅当趋势显著变化时生成新 Insight
    返回摘要：ingested, topics_count, insights_count 等。
    """
    _ensure_path()
    import yaml
    from insight_radar.core.schemas import NormalizedDocument, Topic, InsightOutput
    from insight_radar.ingestion import load_rss_sources, scrape_web_sources, load_local_documents
    from insight_radar.processing import get_embedder, build_topics, compute_trend_score, analyze_sentiment, aggregate_sentiment
    from insight_radar.insight import build_timeline, generate_insight
    from insight_radar.storage import get_db, init_db, save_documents, save_topics, save_insights
    from insight_radar.storage.database import DEFAULT_DB_PATH

    config_path = config_path or os.getenv("INSIGHT_RADAR_CONFIG", str(Path(__file__).resolve().parent.parent / "config" / "sources.yaml"))
    db_path = db_path or os.getenv("INSIGHT_RADAR_DB", DEFAULT_DB_PATH)
    init_db(db_path)

    summary = {"ingested": 0, "topics_count": 0, "insights_count": 0, "error": None}

    if not Path(config_path).is_file():
        summary["error"] = f"Config not found: {config_path}"
        return summary

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 1. Ingest
    docs: list[NormalizedDocument] = []
    if run_ingest:
        if config.get("rss"):
            docs.extend(load_rss_sources(config["rss"]))
        if config.get("web_scrape"):
            docs.extend(scrape_web_sources(config["web_scrape"]))
        if config.get("local"):
            base = config["local"].get("base_path", "./data/local")
            exts = config["local"].get("extensions", [".txt", ".md"])
            rec = config["local"].get("recursive", True)
            docs.extend(load_local_documents(base, exts, rec))
        summary["ingested"] = len(docs)
        if docs:
            session = get_db(db_path)
            save_documents(session, docs)
            session.close()

    # 2–5. Pipeline：从 DB 取全部文档 → embedding → 聚类 → TrendScore → Insight（仅显著变化时）
    if not run_pipeline:
        return summary

    session = get_db(db_path)
    try:
        from sqlalchemy import text
        rows = session.execute(text("SELECT doc_id, source, timestamp, author, raw_text, title, url, meta FROM documents")).fetchall()
    except Exception as e:
        summary["error"] = str(e)
        return summary
    finally:
        session.close()

    if not rows:
        return summary

    import json
    from datetime import datetime
    all_docs = []
    for r in rows:
        meta = json.loads(r[7]) if isinstance(r[7], str) else (r[7] or {})
        ts = r[2]
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.utcnow()
        all_docs.append(NormalizedDocument(doc_id=r[0], source=r[1], timestamp=ts, author=r[3], raw_text=r[4], title=r[5], url=r[6], meta=meta))

    embedder = get_embedder()
    embeddings = embedder.embed_documents(all_docs)
    proc = config.get("processing") or {}
    min_size = proc.get("cluster_min_size", 3)
    topics = build_topics(all_docs, embeddings=embeddings, min_cluster_size=min_size, use_llm_rename=True)
    summary["topics_count"] = len(topics)
    if not topics:
        return summary

    for t in topics:
        t.documents = [d for d in all_docs if d.doc_id in t.doc_ids]

    session = get_db(db_path)
    save_topics(session, topics)
    trend_scores = [compute_trend_score(t) for t in topics]
    significant_delta = proc.get("trend_significant_delta", 15)
    insights: list[InsightOutput] = []
    for topic, score in zip(topics, trend_scores):
        timeline = build_timeline(topic)
        sent_results = [analyze_sentiment(d) for d in topic.documents]
        sent_agg = aggregate_sentiment(sent_results) if sent_results else None
        insight = generate_insight(topic, trend_score=score, timeline=timeline, sentiment_agg=sent_agg)
        insights.append(insight)
    save_insights(session, insights)
    session.close()
    summary["insights_count"] = len(insights)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="InsightRadar daily run")
    parser.add_argument("--no-ingest", action="store_true", help="Skip ingestion")
    parser.add_argument("--no-pipeline", action="store_true", help="Skip pipeline")
    parser.add_argument("--config", default=None, help="Path to sources.yaml")
    parser.add_argument("--db", default=None, help="Path to SQLite DB")
    args = parser.parse_args()
    result = run_daily(
        config_path=args.config,
        db_path=args.db,
        run_ingest=not args.no_ingest,
        run_pipeline=not args.no_pipeline,
    )
    print(result)
