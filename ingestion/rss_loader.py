"""
InsightRadar — RSS 摄取：科技、财经、学术、媒体。
输出统一为 NormalizedDocument。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import feedparser
from pydantic import BaseModel

from ..core.schemas import NormalizedDocument


class RSSSourceConfig(BaseModel):
    name: str
    url: str
    category: str = "general"


def _parse_date(entry: Any) -> datetime:
    """从 RSS entry 解析时间，失败则用当前时间。"""
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = getattr(entry, key, None)
        if parsed:
            try:
                from time import mktime
                return datetime.utcfromtimestamp(mktime(parsed))
            except (TypeError, OSError):
                pass
    return datetime.utcnow()


def _doc_id(source: str, entry: Any) -> str:
    link = getattr(entry, "link", "") or ""
    published = getattr(entry, "published", "") or getattr(entry, "updated", "")
    import hashlib
    raw = f"rss:{source}:{link}:{published}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_rss_feed(url: str, source_name: str, category: str = "general") -> list[NormalizedDocument]:
    """抓取单个 RSS feed，返回 NormalizedDocument 列表。"""
    doc_list: list[NormalizedDocument] = []
    feed = feedparser.parse(url)
    if feed.bozo and not getattr(feed, "entries", None):
        return doc_list
    for entry in feed.entries or []:
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        raw_text = f"{title}\n\n{summary}".strip()
        if not raw_text:
            continue
        doc_list.append(
            NormalizedDocument(
                doc_id=_doc_id(source_name, entry),
                source=f"rss:{source_name}",
                timestamp=_parse_date(entry),
                author=getattr(entry, "author", None),
                raw_text=raw_text[:50000],
                title=title[:2000] if title else None,
                url=getattr(entry, "link", None),
                meta={"category": category},
            )
        )
    return doc_list


def load_rss_sources(sources: list[dict[str, Any]]) -> list[NormalizedDocument]:
    """根据 config/sources.yaml 中 rss 列表抓取，返回合并的文档列表。"""
    out: list[NormalizedDocument] = []
    for s in sources:
        cfg = RSSSourceConfig(**s) if isinstance(s, dict) else s
        if isinstance(cfg, dict):
            cfg = RSSSourceConfig(**cfg)
        docs = load_rss_feed(cfg.url, cfg.name, cfg.category)
        out.extend(docs)
    return out
