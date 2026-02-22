"""
InsightRadar — 新闻网站 HTML 抓取 + 解析。
输出统一为 NormalizedDocument。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from ..core.schemas import NormalizedDocument


class WebScrapeConfig(BaseModel):
    name: str
    url: str
    selector: str = "article"
    category: str = "general"
    max_items: int = 50


def _doc_id(source: str, url: str, title: str, ts: str) -> str:
    import hashlib
    raw = f"web:{source}:{url}:{title}:{ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def scrape_page(
    url: str,
    source_name: str,
    selector: str = "article",
    category: str = "general",
    max_items: int = 50,
) -> list[NormalizedDocument]:
    """抓取单页，按 selector 提取块，转为 NormalizedDocument。"""
    doc_list: list[NormalizedDocument] = []
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "InsightRadar/1.0"})
        resp.raise_for_status()
    except Exception:
        return doc_list
    soup = BeautifulSoup(resp.text, "lxml")
    items = soup.select(selector)[:max_items]
    now = datetime.utcnow().isoformat()
    for i, node in enumerate(items):
        title_node = node.select_one("h1, h2, h3, .title, a")
        title = (title_node.get_text(strip=True) if title_node else "")[:2000]
        body = node.get_text(separator="\n", strip=True)[:50000]
        if not body:
            continue
        link = url
        anchor = node.select_one("a[href]")
        if anchor and anchor.get("href"):
            href = anchor["href"]
            if href.startswith("/"):
                from urllib.parse import urljoin
                link = urljoin(url, href)
            elif href.startswith("http"):
                link = href
        doc_list.append(
            NormalizedDocument(
                doc_id=_doc_id(source_name, link, title, now + str(i)),
                source=f"web:{source_name}",
                timestamp=datetime.utcnow(),
                author=None,
                raw_text=f"{title}\n\n{body}".strip(),
                title=title or None,
                url=link,
                meta={"category": category},
            )
        )
    return doc_list


def scrape_web_sources(sources: list[dict[str, Any]]) -> list[NormalizedDocument]:
    """根据 config 中 web_scrape 列表抓取。"""
    out: list[NormalizedDocument] = []
    for s in sources:
        cfg = WebScrapeConfig(**s) if isinstance(s, dict) else s
        if isinstance(cfg, dict):
            cfg = WebScrapeConfig(**cfg)
        docs = scrape_page(
            cfg.url,
            cfg.name,
            cfg.selector,
            cfg.category,
            cfg.max_items,
        )
        out.extend(docs)
    return out
